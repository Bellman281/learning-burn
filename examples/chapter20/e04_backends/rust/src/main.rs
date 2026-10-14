use std::time::Instant;

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::relu;
use burn::tensor::{Device, Tensor, TensorData};
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

// --- the model: Chapter 13's CNN, weights from Chapter 19's safetensors file ---
#[derive(Module, Debug)]
struct SmallCnn {
    conv1: Conv2d,
    pool: MaxPool2d,
    conv2: Conv2d,
    gap: AdaptiveAvgPool2d,
    head: Linear,
}

impl SmallCnn {
    fn new(device: &Device) -> Self {
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

    fn forward(&self, x: Tensor<4>) -> Tensor<2> {
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

fn load_model(device: &Device) -> SmallCnn {
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
fn images(n: usize, device: &Device) -> Tensor<4> {
    let mut rng = Lcg(7);
    let pixels: Vec<f32> = (0..n).flat_map(|i| make_image(i % 4, &mut rng)).collect();
    Tensor::from_data(TensorData::new(pixels, [n, 1, SIZE, SIZE]), device)
}

// Written once, for whichever device it is given (see Chapter 21). Returns (logits, median ms).
fn run(device: &Device, n: usize) -> (Vec<f32>, f64) {
    let model = load_model(device);
    let x = images(n, device);
    let logits: Vec<f32> = model.forward(x.clone()).into_data().try_to_vec().unwrap(); // warm-up too
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

/// Returns (rows of (backend, ms for 256 images, max |diff vs Flex|)).
fn build() -> Vec<(&'static str, f64, f32)> {
    let n = 256;
    let (reference, ms_flex) = run(&Device::flex(), n);
    #[allow(deprecated)] // NdArray is deprecated in 0.22: kept as the baseline it replaces
    let (nd, ms_nd) = run(&Device::ndarray(), n);
    #[allow(unused_mut)]
    let mut rows = vec![
        ("Flex", ms_flex, 0.0),
        ("NdArray", ms_nd, max_diff(&nd, &reference)),
    ];
    // A GPU: `cargo run --release --features gpu` (Metal on a Mac, Vulkan/DX12 elsewhere).
    #[cfg(feature = "gpu")]
    {
        let gpu_device = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
        let (gpu, ms_gpu) = run(&gpu_device, n);
        rows.push(("Wgpu", ms_gpu, max_diff(&gpu, &reference)));
    }
    rows
}

fn main() {
    println!("the same forward pass, 256 images, on each backend");
    println!("backend   ms / batch   max |diff vs Flex|");
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
