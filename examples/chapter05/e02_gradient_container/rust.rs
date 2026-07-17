use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;

// Unlike PyTorch (which stores each gradient on the tensor as `.grad`), Burn's
// `backward()` RETURNS all gradients in a container, and you look each one up.

/// Returns (loss, dL/da, dL/db) for L = sum(a*b): dL/da = b, dL/db = a.
fn build() -> (f32, Vec<f32>, Vec<f32>) {
    let device = Default::default();

    let a = Tensor::<Backend, 1>::from_floats([2.0, 3.0], &device).require_grad();
    let b = Tensor::<Backend, 1>::from_floats([4.0, 5.0], &device).require_grad();

    let loss = (a.clone() * b.clone()).sum();
    let grads = loss.backward();

    let da = a.grad(&grads).unwrap(); // = b = [4, 5]
    let db = b.grad(&grads).unwrap(); // = a = [2, 3]

    (
        loss.into_scalar(),
        da.into_data().to_vec().unwrap(),
        db.into_data().to_vec().unwrap(),
    )
}

fn main() {
    let (loss, da, db) = build();
    println!("loss  = {loss}");
    println!("dL/da = {da:?}");
    println!("dL/db = {db:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (loss, da, db) = build();
        assert!((loss - 23.0).abs() < 1e-6, "loss = {loss}");
        assert_eq!(da, vec![4.0, 5.0]); // = b
        assert_eq!(db, vec![2.0, 3.0]); // = a
    }
}
