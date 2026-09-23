use std::time::Instant;

use burn::backend::{Flex, NdArray};
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

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

// n test images (cycling through the classes), as one tensor [n, 1, 12, 12].
fn images<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 4> {
    let mut rng = Lcg(7);
    let pixels: Vec<f32> = (0..n).flat_map(|i| make_image(i % 4, &mut rng)).collect();
    Tensor::from_data(TensorData::new(pixels, [n, 1, SIZE, SIZE]), device)
}

// Written once, generic over the backend (Chapter 14). Returns (logits, median ms).
fn run<B: Backend>(n: usize) -> (Vec<f32>, f64) {
    let device = Default::default();
    let model = load_model::<B>(&device);
    let x = images::<B>(n, &device);
    let logits: Vec<f32> = model.forward(x.clone()).into_data().to_vec().unwrap(); // warm-up too
    let mut times: Vec<f64> = (0..30)
        .map(|_| {
            let t = Instant::now();
            let _ = model.forward(x.clone()).into_data();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    (logits, times[times.len() / 2])
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Returns (rows of (backend, ms for 256 images, max |diff vs NdArray|)).
fn build() -> Vec<(&'static str, f64, f32)> {
    let n = 256;
    let (reference, ms_nd) = run::<NdArray>(n);
    let (flex, ms_flex) = run::<Flex>(n);
    #[allow(unused_mut)]
    let mut rows = vec![
        ("NdArray", ms_nd, 0.0),
        ("Flex", ms_flex, max_diff(&flex, &reference)),
    ];
    // A GPU: `cargo run --release --features gpu` (Metal on a Mac, Vulkan/DX12 elsewhere).
    #[cfg(feature = "gpu")]
    {
        let (gpu, ms_gpu) = run::<burn::backend::Wgpu>(n);
        rows.push(("Wgpu", ms_gpu, max_diff(&gpu, &reference)));
    }
    rows
}

fn main() {
    println!("the same forward pass, 256 images, on each backend");
    println!("backend   ms / batch   max |diff vs NdArray|");
    for (name, ms, diff) in build() {
        println!("{name:<8}  {ms:>10.3}   {diff:e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backends_agree() {
        for (name, _, diff) in build() {
            assert!(diff < 1e-4, "{name}: {diff}");
        }
    }
}
