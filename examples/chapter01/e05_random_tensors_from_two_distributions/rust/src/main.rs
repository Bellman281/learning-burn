use burn::tensor::{Device, Distribution, Tensor};

fn main() {
    let device = Device::flex();

    // uniform on [0, 1)
    let u = Tensor::<2>::random([2, 3], Distribution::Default, &device);

    // standard normal (mean 0, std 1)
    let n = Tensor::<2>::random([2, 3], Distribution::Normal(0.0, 1.0), &device);

    println!("uniform = {}", u);
    println!("normal = {}", n);
}
