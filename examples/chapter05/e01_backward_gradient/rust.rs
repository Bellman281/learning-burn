use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

// Autodiff is a backend *decorator*: wrap any backend to get gradients.
type Backend = Autodiff<NdArray>;

/// Returns (f, df/dx). f(x) = sum(x^2); by hand df/dx = 2x.
fn build() -> (f32, Vec<f32>) {
    let device = Default::default();

    let x = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device).require_grad();
    let f = (x.clone() * x.clone()).sum(); // 1 + 4 + 9 = 14

    let grads = f.backward();
    let dx = x.grad(&grads).unwrap(); // [2, 4, 6] == 2x

    (f.into_scalar(), dx.into_data().to_vec().unwrap())
}

fn main() {
    let (f, dx) = build();
    println!("f  = {f}");
    println!("dx = {dx:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (f, dx) = build();
        assert!((f - 14.0).abs() < 1e-6, "f = {f}");
        assert_eq!(dx, vec![2.0, 4.0, 6.0]); // 2x, exactly PyTorch's x.grad
    }
}
