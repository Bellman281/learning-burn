use burn::tensor::{Device, Tensor, TensorData};

struct BodyMetrics {
    age: i8,
    height: i16,
    weight: f32,
}

fn main() {
    let device = Device::flex();

    let bmi = BodyMetrics {
        age: 25,
        height: 180,
        weight: 80.0,
    };

    let data = TensorData::from([bmi.age as f32, bmi.height as f32, bmi.weight]);
    let t = Tensor::<1>::from_data(data, &device);

    println!("t = {}", t);
}
