use burn::linalg;
use burn::tensor::{Device, Tensor};

fn main() {
    let dev = Device::flex();

    let m = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev);
    let eye = Tensor::<2>::eye(3, &dev); // identity

    let lo = m.clone().tril(0); // lower-triangular part
    let up = m.clone().triu(0); // upper-triangular part
    let tr: Tensor<1> = linalg::trace(m); // sum of the diagonal

    println!("identity     = {}", eye.to_data());
    println!("lo           = {}", lo.to_data());
    println!("up           = {}", up.to_data());
    println!("trace        = {}", tr.to_data());
}
