use burn::tensor::{Device, Tensor};

// There is no `no_grad` block in Burn. For inference you either create tensors
// on a device without autodiff, or call `inner()` (an alias of
// `without_autodiff()`) to take a tensor off autodiff and skip graph tracking.
fn main() {
    let dev = Device::flex().autodiff();

    let x = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &dev).require_grad();

    // Drop autodiff -> a plain flex tensor, no tracking.
    let plain = x.inner();
    let y = plain.add_scalar(5.0); // cheap inference op

    println!("{}", y.to_data()); // [6.0, 7.0, 8.0]
}
