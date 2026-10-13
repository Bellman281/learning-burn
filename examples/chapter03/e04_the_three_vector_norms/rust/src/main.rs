use burn::linalg;
use burn::tensor::{Device, Tensor};

fn main() {
    let dev = Device::flex();
    let x = Tensor::<1>::from_floats([3.0, 4.0], &dev);

    let l2 = linalg::l2_norm(x.clone(), 0); // 5.0 — the usual one
    let l1 = linalg::l1_norm(x.clone(), 0); // 7.0
    let linf = linalg::max_abs_norm(x, 0); // 4.0

    println!("L2 norm: {}", l2.to_data());
    println!("L1 norm: {}", l1.to_data());
    println!("L∞ norm: {}", linf.to_data());
}
