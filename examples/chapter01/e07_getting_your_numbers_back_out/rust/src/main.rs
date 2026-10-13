use burn::tensor::{Device, Tensor};

fn main() {
    let device = Device::flex();
    let t = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device);

    // inspect without consuming
    let data = t.clone().to_data();
    let values: &[f32] = data.as_slice().unwrap();
    println!("values (borrowed view) = {:?}", values);

    // or consume it when you are done
    let owned = t.into_data();
    println!("owned data = {:?}", owned);
}
