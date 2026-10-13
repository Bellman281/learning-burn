use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::{Device, Int, Tensor, TensorData};

fn main() {
    let dev = Device::flex();

    // logits: [batch, num_classes]; cross-entropy takes raw logits (not softmaxed).
    let logits = Tensor::<2>::from_floats([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], &dev);

    // targets: the correct class index per row.
    let targets = Tensor::<1, Int>::from_data(TensorData::from([0i64, 1]), &dev);

    let loss_fn = CrossEntropyLossConfig::new().init(&dev);
    let loss = loss_fn.forward(logits, targets);

    println!("cross-entropy = {}", loss.to_data()); // ~0.319
}
