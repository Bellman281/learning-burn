use burn::tensor::{Device, Tensor};

fn main() {
    let dev = Device::flex();

    // [m, k] times [k, n] -> [m, n]; the inner dimensions must match.
    let a = Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &dev); // [2, 3]
    let b = Tensor::<2>::from_floats([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], &dev); // [3, 2]

    // Matrix product (element-wise `*` would NOT be matmul).
    let c = a.matmul(b); // [2, 2]
    println!("{}", c.to_data());
}
