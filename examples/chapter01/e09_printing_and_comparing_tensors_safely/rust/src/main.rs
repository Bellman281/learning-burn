use burn::tensor::{Device, Tensor, check_closeness};

fn main() {
    let dev = Device::flex();

    let tensor = Tensor::<2>::full([2, 3], 0.123456789, &dev);
    println!("{}", tensor);

    // control precision with standard Rust formatting
    println!("{:.2}", tensor);

    let a = Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &dev);
    let b = Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.001], &dev);

    check_closeness(&a, &b);
}
