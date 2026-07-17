use burn::backend::NdArray;
use burn::tensor::{Tensor, linalg};

type Backend = NdArray;

fn build() -> (
    Tensor<Backend, 2>,
    Tensor<Backend, 1>,
    Tensor<Backend, 1>,
    Tensor<Backend, 2>,
) {
    let device = Default::default();

    let m = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let v = Tensor::<Backend, 1>::from_floats([1.0, 1.0], &device);

    let mt = m.clone().transpose(); // or .t()
    let mv = linalg::matvec(m.clone(), v.clone()); // matrix * vector -> [2]
    let d = v.clone().dot(v.clone()); // scalar dot product
    let op: Tensor<Backend, 2> = linalg::outer(v.clone(), v); // outer product -> [2, 2]

    (mt, mv, d, op)
}

fn main() {
    let (mt, mv, d, op) = build();
    println!("transpose =\n{}", mt.to_data());
    println!("matvec    = {}", mv.to_data());
    println!("dot       = {}", d.to_data());
    println!("outer     =\n{}", op.to_data());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (mt, mv, d, op) = build();

        assert_eq!(mt.dims(), [2, 2]);
        assert_eq!(mt.into_data().to_vec::<f32>().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);

        assert_eq!(mv.into_data().to_vec::<f32>().unwrap(), vec![3.0, 7.0]);

        assert_eq!(d.into_scalar(), 2.0);

        assert_eq!(op.dims(), [2, 2]);
        assert_eq!(op.into_data().to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    }
}
