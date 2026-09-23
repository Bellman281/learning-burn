use std::sync::Arc;

use axum::Router;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use serde::{Deserialize, Serialize};

const SIZE: usize = 12;

// --- the data: 12x12 "shapes", generated in code (identical in python.py) ---
// A tiny LCG so Rust and Python draw exactly the same "random" numbers.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    fn below(&mut self, n: u32) -> usize {
        (self.next() % n) as usize
    }
}

// class 0 = horizontal bar, 1 = vertical bar, 2 = diagonal, 3 = plus sign
fn make_image(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut img = vec![0.0f32; SIZE * SIZE];
    match class {
        0 => {
            let (r, c) = (rng.below(12), rng.below(7));
            (0..6).for_each(|j| img[r * SIZE + c + j] = 1.0);
        }
        1 => {
            let (c, r) = (rng.below(12), rng.below(7));
            (0..6).for_each(|i| img[(r + i) * SIZE + c] = 1.0);
        }
        2 => {
            let (r, c) = (rng.below(7), rng.below(7));
            (0..6).for_each(|i| img[(r + i) * SIZE + c + i] = 1.0);
        }
        _ => {
            let (r, c) = (2 + rng.below(8), 2 + rng.below(8));
            for d in 0..5 {
                img[r * SIZE + c + d - 2] = 1.0;
                img[(r + d - 2) * SIZE + c] = 1.0;
            }
        }
    }
    // background noise in [0, 0.25): exact in f32, so both languages agree bit for bit
    img.iter_mut()
        .for_each(|p| *p += rng.next() as f32 / 32768.0 * 0.25);
    img
}

// --- the model: Chapter 15's CNN, weights from Chapter 19's safetensors file ---
#[derive(Module, Debug)]
struct SmallCnn<B: Backend> {
    conv1: Conv2d<B>,
    pool: MaxPool2d,
    conv2: Conv2d<B>,
    gap: AdaptiveAvgPool2d,
    head: Linear<B>,
}

impl<B: Backend> SmallCnn<B> {
    fn new(device: &B::Device) -> Self {
        let same = PaddingConfig2d::Explicit(1, 1, 1, 1);
        Self {
            conv1: Conv2dConfig::new([1, 4], [3, 3])
                .with_padding(same.clone())
                .init(device),
            pool: MaxPool2dConfig::new([2, 2]).init(),
            conv2: Conv2dConfig::new([4, 8], [3, 3])
                .with_padding(same)
                .init(device),
            gap: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            head: LinearConfig::new(8, 4).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.pool.forward(relu(self.conv1.forward(x)));
        let x = relu(self.conv2.forward(x));
        let [batch, channels, _, _] = x.dims();
        self.head
            .forward(self.gap.forward(x).reshape([batch, channels]))
    }
}

const WEIGHTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../chapter19/weights/small_cnn.safetensors"
);

fn load_model<B: Backend>(device: &B::Device) -> SmallCnn<B> {
    let mut store = SafetensorsStore::from_file(WEIGHTS)
        .with_from_adapter(PyTorchToBurnAdapter)
        .with_key_remapping(r"^features\.0\.(.+)$", "conv1.$1")
        .with_key_remapping(r"^features\.3\.(.+)$", "conv2.$1")
        .with_key_remapping(r"^classifier\.(.+)$", "head.$1");
    let mut model = SmallCnn::new(device);
    model.load_from(&mut store).expect("load weights");
    model
}

// --- the service ---
#[derive(Deserialize, Serialize)]
struct PredictRequest {
    pixels: Vec<f32>, // one 12x12 image, row by row
}

#[derive(Deserialize, Serialize, Debug)]
struct PredictResponse {
    class: usize,
    logits: Vec<f32>,
}

type Shared = Arc<SmallCnn<NdArray>>; // loaded once, shared by every request

async fn predict(
    State(model): State<Shared>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, String)> {
    if req.pixels.len() != SIZE * SIZE {
        let msg = format!("expected {} pixels, got {}", SIZE * SIZE, req.pixels.len());
        return Err((StatusCode::UNPROCESSABLE_ENTITY, msg));
    }
    let device = Default::default();
    let x =
        Tensor::<NdArray, 4>::from_data(TensorData::new(req.pixels, [1, 1, SIZE, SIZE]), &device);
    let logits: Vec<f32> = model.forward(x).into_data().to_vec().unwrap();
    let class = (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap();
    Ok(Json(PredictResponse { class, logits }))
}

fn app(model: Shared) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/predict", post(predict))
        .with_state(model)
}

/// Starts the server on a free port, then acts as its own client.
/// Returns (health, classes for the 8 test images, status for a bad request).
async fn build() -> (String, Vec<usize>, u16) {
    let model: Shared = Arc::new(load_model::<NdArray>(&Default::default()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app(model)).await.unwrap() });

    let client = reqwest::Client::new();
    let health = client
        .get(format!("{url}/health"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let mut rng = Lcg(7); // Chapter 15's test images
    let mut classes = Vec::new();
    for i in 0..8 {
        let body = PredictRequest {
            pixels: make_image(i % 4, &mut rng),
        };
        let resp: PredictResponse = client
            .post(format!("{url}/predict"))
            .json(&body)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if i == 0 {
            println!("POST /predict (image 0) -> {resp:?}");
        }
        classes.push(resp.class);
    }

    let bad = client
        .post(format!("{url}/predict"))
        .json(&PredictRequest {
            pixels: vec![0.0; 10],
        })
        .send()
        .await
        .unwrap();
    (health, classes, bad.status().as_u16())
}

#[tokio::main]
async fn main() {
    let (health, classes, bad) = build().await;
    println!("GET /health -> {health}");
    println!("classes for the 8 test images = {classes:?}");
    println!("a request with 10 pixels -> HTTP {bad}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn matches_pytorch() {
        let (health, classes, bad) = build().await;
        assert_eq!(health, "ok");
        assert_eq!(classes, vec![0, 1, 2, 3, 0, 1, 2, 3]); // python/main.py serves the same
        assert_eq!(bad, 422);
    }
}
