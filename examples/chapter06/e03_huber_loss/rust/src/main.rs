use burn::nn::loss::{HuberLossConfig, Reduction};
use burn::tensor::{Device, Tensor};

fn main() {
    let dev = Device::flex();

    let pred = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev);
    let target = Tensor::<2>::from_floats([[2.0, 1.0], [3.0, 2.0]], &dev);

    let loss = HuberLossConfig::new(1.0) // delta = 1.0
        .init()
        .forward(pred, target, Reduction::Mean);

    println!("loss = {}", loss.to_data()); // 0.625
}
