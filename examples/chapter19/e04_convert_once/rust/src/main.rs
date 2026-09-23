use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_store::{ModuleSnapshot, PytorchStore};

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

// --- the Burn model: Chapter 15's CNN, with Burn-style field names ---
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

// The first 8 test images of Chapter 15 (seed 7), as one batch.
fn test_images<B: Backend>(device: &B::Device) -> Tensor<B, 4> {
    let mut rng = Lcg(7);
    let pixels: Vec<f32> = (0..8).flat_map(|i| make_image(i % 4, &mut rng)).collect();
    Tensor::from_data(TensorData::new(pixels, [8, 1, SIZE, SIZE]), device)
}

// PyTorch's logits for the same 8 images, written by weights/make_weights.py.
fn reference() -> Vec<f32> {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../weights/reference.json"
    ))
    .unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    json["logits"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
        })
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

const PT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/small_cnn.pt");

/// The one-off converter: PyTorch file in, Burn record out.
fn convert(out: &std::path::Path) -> SmallCnn<NdArray> {
    let device = Default::default();
    let mut store = PytorchStore::from_file(PT)
        .with_key_remapping(r"^features\.0\.(.+)$", "conv1.$1")
        .with_key_remapping(r"^features\.3\.(.+)$", "conv2.$1")
        .with_key_remapping(r"^classifier\.(.+)$", "head.$1");
    let mut model = SmallCnn::<NdArray>::new(&device);
    model.load_from(&mut store).expect("load PyTorch weights");
    model
        .clone()
        .save_file(out, &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
        .expect("save Burn record");
    model
}

/// What the serving program does from then on: only `burn`, no importer.
fn load(path: &std::path::Path) -> SmallCnn<NdArray> {
    let device = Default::default();
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(path.to_path_buf(), &device)
        .expect("load Burn record");
    SmallCnn::new(&device).load_record(record)
}

/// Returns (file sizes, logits identical after the round trip, max |Burn - PyTorch|).
fn build() -> (Vec<(String, u64)>, bool, f32) {
    let device = Default::default();
    let out = std::env::temp_dir().join("learning-burn-c19e4-small_cnn");
    let converted = convert(&out);
    let served = load(&out.with_extension("mpk"));

    let a: Vec<f32> = converted
        .forward(test_images(&device))
        .into_data()
        .to_vec()
        .unwrap();
    let b: Vec<f32> = served
        .forward(test_images(&device))
        .into_data()
        .to_vec()
        .unwrap();

    let sizes = [
        PT.to_string(),
        out.with_extension("mpk").display().to_string(),
    ]
    .into_iter()
    .map(|p| {
        let name = std::path::Path::new(&p)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        (name, std::fs::metadata(&p).unwrap().len())
    })
    .collect();
    (sizes, a == b, max_diff(&b, &reference()))
}

fn main() {
    let (sizes, same, diff) = build();
    for (name, bytes) in &sizes {
        println!("{name:<34} {bytes:>5} bytes");
    }
    println!("logits identical after the round trip: {same}");
    println!("max |Burn - torch| = {diff:e}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (_, same, diff) = build();
        assert!(same);
        assert!(diff < 1e-5, "{diff}");
    }
}
