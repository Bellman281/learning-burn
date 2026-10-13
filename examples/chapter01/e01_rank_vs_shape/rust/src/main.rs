use burn::tensor::{Device, Tensor};

fn main() {
    let floats = [1.0, 2.0, 3.0, 4.0, 5.0];
    let device = Device::flex();

    // CORRECT: a 1-dimensional tensor that happens to hold 5 elements
    let tensor = Tensor::<1>::from_floats(floats, &device);

    println!("tensor = {}", tensor);

    // WRONG: this asks for a 5-dimensional tensor. It compiles, then panics at run time:
    // five numbers in a row have the rank-1 shape [5], not five axes.
    // let tensor = Tensor::<5>::from_floats(floats, &device);
}
