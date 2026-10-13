use burn::tensor::{Device, Tensor, TensorData};

fn main() {
    let device = Device::flex();
    // from_floats: most ergonomic for f32 data
    let a = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device);

    // from_data: the general path, via a TensorData struct
    let b = Tensor::<1>::from_data(TensorData::from([1.0, 2.0, 3.0]), &device);

    // a 2-D tensor from nested arrays
    let m = Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

    println!("a = {}", a);
    println!("b = {}", b);
    println!("m = {}", m);
}
