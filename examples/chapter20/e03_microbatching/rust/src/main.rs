use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::{Json, State};
use axum::routing::post;
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
use tokio::sync::{mpsc, oneshot};

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

#[derive(Deserialize, Serialize)]
struct PredictRequest {
    pixels: Vec<f32>,
}

#[derive(Deserialize, Serialize)]
struct PredictResponse {
    class: usize,
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len())
        .max_by(|&a, &b| row[a].total_cmp(&row[b]))
        .unwrap()
}

// ---- strategy A: every request runs its own forward pass (on a blocking thread) ----
async fn predict_direct(
    State(model): State<Arc<SmallCnn<NdArray>>>,
    Json(req): Json<PredictRequest>,
) -> Json<PredictResponse> {
    let class = tokio::task::spawn_blocking(move || {
        let x = Tensor::from_data(
            TensorData::new(req.pixels, [1, 1, SIZE, SIZE]),
            &Default::default(),
        );
        argmax(&model.forward(x).into_data().to_vec::<f32>().unwrap())
    })
    .await
    .unwrap();
    Json(PredictResponse { class })
}

// ---- strategy B: requests queue up; one worker runs them in batches ----
type Job = (Vec<f32>, oneshot::Sender<usize>);

fn spawn_batcher(
    model: SmallCnn<NdArray>,
    max_batch: usize,
    max_wait: Duration,
) -> mpsc::Sender<Job> {
    let (tx, mut rx) = mpsc::channel::<Job>(1024);
    std::thread::spawn(move || {
        let device = Default::default();
        // Block until the first request arrives, then collect more for at most max_wait.
        while let Some(first) = rx.blocking_recv() {
            let mut jobs = vec![first];
            let deadline = Instant::now() + max_wait;
            while jobs.len() < max_batch && Instant::now() < deadline {
                match rx.try_recv() {
                    Ok(job) => jobs.push(job),
                    Err(_) => std::thread::yield_now(),
                }
            }
            let n = jobs.len();
            let pixels: Vec<f32> = jobs.iter().flat_map(|(p, _)| p.iter().copied()).collect();
            let x = Tensor::<NdArray, 4>::from_data(
                TensorData::new(pixels, [n, 1, SIZE, SIZE]),
                &device,
            );
            let logits: Vec<f32> = model.forward(x).into_data().to_vec().unwrap();
            for ((_, reply), row) in jobs.into_iter().zip(logits.chunks(4)) {
                let _ = reply.send(argmax(row));
            }
            BATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });
    tx
}
static BATCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

async fn predict_batched(
    State(queue): State<mpsc::Sender<Job>>,
    Json(req): Json<PredictRequest>,
) -> Json<PredictResponse> {
    let (tx, rx) = oneshot::channel();
    queue.send((req.pixels, tx)).await.unwrap();
    Json(PredictResponse {
        class: rx.await.unwrap(),
    })
}

// ---- the load test: `clients` concurrent clients, `per_client` requests each ----
struct Report {
    requests: usize,
    seconds: f64,
    p50_ms: f64,
    p99_ms: f64,
    correct: usize,
}

async fn load_test(app: Router, clients: usize, per_client: usize) -> Report {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/predict", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let start = Instant::now();
    let tasks: Vec<_> = (0..clients)
        .map(|c| {
            let url = url.clone();
            tokio::spawn(async move {
                let client = reqwest::Client::new();
                let mut rng = Lcg(7 + c as u32);
                let mut out = Vec::new();
                for i in 0..per_client {
                    let class = i % 4;
                    let body = PredictRequest {
                        pixels: make_image(class, &mut rng),
                    };
                    let t = Instant::now();
                    let resp: PredictResponse = client
                        .post(&url)
                        .json(&body)
                        .send()
                        .await
                        .unwrap()
                        .json()
                        .await
                        .unwrap();
                    out.push((t.elapsed().as_secs_f64() * 1e3, resp.class == class));
                }
                out
            })
        })
        .collect();
    let mut latencies = Vec::new();
    let mut correct = 0;
    for t in tasks {
        for (ms, ok) in t.await.unwrap() {
            latencies.push(ms);
            correct += ok as usize;
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    latencies.sort_by(f64::total_cmp);
    let pct = |p: f64| latencies[((latencies.len() - 1) as f64 * p) as usize];
    Report {
        requests: latencies.len(),
        seconds,
        p50_ms: pct(0.50),
        p99_ms: pct(0.99),
        correct,
    }
}

/// Returns (direct report, batched report, mean batch size).
async fn build() -> (Report, Report, f64) {
    let device = Default::default();
    let (clients, per_client) = (32, 40);

    let direct = Router::new()
        .route("/predict", post(predict_direct))
        .with_state(Arc::new(load_model::<NdArray>(&device)));
    let a = load_test(direct, clients, per_client).await;

    let queue = spawn_batcher(load_model::<NdArray>(&device), 32, Duration::from_millis(2));
    let batched = Router::new()
        .route("/predict", post(predict_batched))
        .with_state(queue);
    let b = load_test(batched, clients, per_client).await;
    let mean_batch = b.requests as f64 / BATCHES.load(std::sync::atomic::Ordering::Relaxed) as f64;
    (a, b, mean_batch)
}

#[tokio::main]
async fn main() {
    let (a, b, mean_batch) = build().await;
    println!("32 clients x 40 requests each");
    println!("strategy   requests/s  p50 ms  p99 ms  correct");
    for (name, r) in [("direct", &a), ("batched", &b)] {
        println!(
            "{name:<9}  {:>10.0}  {:>6.2}  {:>6.2}  {}/{}",
            r.requests as f64 / r.seconds,
            r.p50_ms,
            r.p99_ms,
            r.correct,
            r.requests
        );
    }
    println!("mean batch size in the batched run = {mean_batch:.1}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn both_strategies_answer_correctly() {
        let (a, b, mean_batch) = build().await;
        assert_eq!((a.correct, a.requests), (1280, 1280));
        assert_eq!((b.correct, b.requests), (1280, 1280));
        assert!(mean_batch >= 1.0);
    }
}
