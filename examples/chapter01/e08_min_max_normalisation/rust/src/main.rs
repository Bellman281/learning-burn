use burn::tensor::{Device, Tensor};

fn main() {
    let device = Device::flex();

    let input = Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);

    let min = input.clone().min();
    let max = input.clone().max();
    let input = (input.clone() - min.clone()).div(max - min);

    println!("{}", input.to_data());
    // [0.0, 0.33333334, 0.6666667, 1.0]
}
