use burn::backend::NdArray;
use burn::tensor::{Tensor, linalg};

type B = NdArray;

fn main() {
    let dev = Default::default();

    let x = Tensor::<B, 1>::from_floats([3.0, 4.0], &dev);
    let n2 = linalg::l2_norm(x.clone(), 0); // 5.0

    let unit = linalg::vector_normalize(x, 2.0, 0, 1e-12);
    let m = Tensor::<B, 2>::from_floats([[4.0, 3.0], [6.0, 3.0]], &dev);
    let d = linalg::det::<B, 2, 2, 1>(m); // -6.0

    println!("n2   = {}", n2.to_data());
    println!("d   = {}", d.to_data());
    println!("unit = {}", unit.to_data());
}
