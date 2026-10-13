use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::tensor::{Device, Int, Tensor};

fn main() {
    let dev = Device::flex();

    let logits = Tensor::<1>::from_floats([2.0, -1.5, 0.3], &dev);
    let targets = Tensor::<1, Int>::from_ints([1, 0, 1], &dev);

    let loss = BinaryCrossEntropyLossConfig::new()
        .with_logits(true) // inputs are logits; sigmoid applied inside
        .init(&dev)
        .forward(logits, targets);

    println!("loss = {}", loss.to_data()); // ~0.294
}
