use burn::backend::NdArray;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

// The model is NOT written here. build.rs generated it from the ONNX file.
#[allow(dead_code)] // generated code includes loaders this example does not use
mod small_cnn {
    include!(concat!(env!("OUT_DIR"), "/model/small_cnn.rs"));
}
use small_cnn::Model;

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

/// Returns (generated source lines, weights file bytes, predictions, max |Burn - PyTorch|).
fn build() -> (usize, u64, Vec<i64>, f32) {
    let device = Default::default();
    let generated = concat!(env!("OUT_DIR"), "/model/small_cnn.rs");
    let weights = concat!(env!("OUT_DIR"), "/model/small_cnn.bpk");
    let lines = std::fs::read_to_string(generated).unwrap().lines().count();
    let bytes = std::fs::metadata(weights).unwrap().len();

    let model: Model<NdArray> = Model::from_file(weights, &device);
    let logits = model.forward(test_images(&device));
    let predicted: Vec<i64> = logits.clone().argmax(1).into_data().to_vec().unwrap();
    let values: Vec<f32> = logits.into_data().to_vec().unwrap();
    (lines, bytes, predicted, max_diff(&values, &reference()))
}

fn main() {
    let (lines, bytes, predicted, diff) = build();
    println!("build.rs generated small_cnn.rs ({lines} lines) and small_cnn.bpk ({bytes} bytes)");
    println!("predictions        = {predicted:?}");
    println!("max |Burn - torch| = {diff:e}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (_, _, predicted, diff) = build();
        assert_eq!(predicted, vec![0, 1, 2, 3, 0, 1, 2, 3]);
        assert!(diff < 1e-5, "{diff}");
    }
}
