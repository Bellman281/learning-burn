use burn::tensor::{Device, Tensor};
// Unlike PyTorch (which stores each gradient on the tensor as `.grad`), Burn's
// `backward()` RETURNS all gradients in a container, and you look each one up.
fn main() {
    let dev = Device::flex().autodiff();

    let a = Tensor::<1>::from_floats([2.0, 3.0], &dev).require_grad();
    let b = Tensor::<1>::from_floats([4.0, 5.0], &dev).require_grad();

    // L = sum(a * b) -> dL/da = b, dL/db = a

    let loss = (a.clone() * b.clone()).sum();
    let grads = loss.backward();
    let da = a.grad(&grads).unwrap(); // = b = [4, 5]
    let db = b.grad(&grads).unwrap(); // = a = [2, 3]

    println!("loss  = {}", loss.to_data()); // [23.0]
    println!("dL/da = {}", da.to_data()); // [4.0, 5.0]
    println!("dL/db = {}", db.to_data()); // [2.0, 3.0]
}
