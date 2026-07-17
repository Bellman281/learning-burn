use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn build() -> Tensor<Backend, 2> {
    let device = Default::default();

    // [m, k] times [k, n] -> [m, n]; the inner dimensions must match.
    let a = Tensor::<Backend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device); // [2, 3]
    let b = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], &device); // [3, 2]

    a.matmul(b) // [2, 2]  (element-wise `*` would NOT be matmul)
}

fn main() {
    println!("{}", build().to_data());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let c = build();
        assert_eq!(c.dims(), [2, 2]);
        assert_eq!(
            c.into_data().to_vec::<f32>().unwrap(),
            vec![4.0, 5.0, 10.0, 11.0]
        );
    }
}
