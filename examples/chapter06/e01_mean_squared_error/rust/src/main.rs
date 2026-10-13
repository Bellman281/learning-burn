use burn::nn::loss::{MseLoss, Reduction};
use burn::tensor::{Device, Tensor};

fn main() {
    let dev = Device::flex();

    let pred = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev);
    let target = Tensor::<2>::from_floats([[2.0, 1.0], [3.0, 2.0]], &dev);

    let mse = MseLoss::new();
    let loss = mse.forward(pred.clone(), target.clone(), Reduction::Mean);

    println!("MSE = {}", loss.to_data()); // mean of [1, 1, 0, 4] = 1.5
}
