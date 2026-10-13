use burn::tensor::{Device, Tensor};

fn main() {
    // Autodiff is a property of the *device*: ask for it, and tensors made there
    // can record the operations they take part in.
    let dev = Device::flex().autodiff();

    // f(x) = sum(x^2). By hand, df/dx = 2x.
    let x = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &dev).require_grad();
    let f = (x.clone() * x.clone()).sum(); // 1 + 4 + 9 = 14

    // One call computes the gradient for us.
    let grads = f.backward();
    let dx = x.grad(&grads).unwrap(); // [2, 4, 6] == 2x

    println!("f  = {}", f.to_data()); // [14.0]
    println!("dx = {}", dx.to_data()); // [2.0, 4.0, 6.0]
}
