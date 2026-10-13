use burn::tensor::{Device, Tensor};

fn main() {
    let dev = Device::flex();

    let x = Tensor::<1>::from_floats([-1.0, 0.0, 2.0], &dev);

    let y = x.clone().exp().log1p(); // chain transcendental ops
    println!("y      = {}", y.to_data());

    let c = x.clone().clamp(0.0, 1.0); // torch.clamp
    println!("c      = {}", c.to_data());

    let mask = x.clone().greater_elem(0.0); // Bool: [false, false, true]
    println!("mask   = {}", mask.to_data());

    let picked = x.mask_where(mask, Tensor::<1>::from_floats([9.0, 9.0, 9.0], &dev));
    println!("picked = {}", picked.to_data());
}
