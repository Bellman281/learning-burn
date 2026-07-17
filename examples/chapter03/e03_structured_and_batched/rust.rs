use burn::backend::NdArray;
use burn::tensor::{Tensor, linalg};

type Backend = NdArray;

fn build() -> (
    Tensor<Backend, 2>,
    Tensor<Backend, 2>,
    Tensor<Backend, 2>,
    Tensor<Backend, 1>,
    Tensor<Backend, 3>,
) {
    let device = Default::default();

    let m = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let eye = Tensor::<Backend, 2>::eye(3, &device);

    let lo = m.clone().tril(0); // lower-triangular part
    let up = m.clone().triu(0); // upper-triangular part
    let tr: Tensor<Backend, 1> = linalg::trace(m); // sum of the diagonal

    // Batched matmul: 8 matrices [2,3] times 8 matrices [3,4] -> [8, 2, 4]
    let ba = Tensor::<Backend, 3>::zeros([8, 2, 3], &device);
    let bb = Tensor::<Backend, 3>::zeros([8, 3, 4], &device);
    let bc = ba.matmul(bb);

    (eye, lo, up, tr, bc)
}

fn main() {
    let (eye, lo, up, tr, bc) = build();
    println!("identity =\n{}", eye.to_data());
    println!("lo       = {}", lo.to_data());
    println!("up       = {}", up.to_data());
    println!("trace    = {}", tr.to_data());
    println!("batched dims = {:?}", bc.dims());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (eye, lo, up, tr, bc) = build();

        assert_eq!(
            eye.into_data().to_vec::<f32>().unwrap(),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(lo.into_data().to_vec::<f32>().unwrap(), vec![1.0, 0.0, 3.0, 4.0]);
        assert_eq!(up.into_data().to_vec::<f32>().unwrap(), vec![1.0, 2.0, 0.0, 4.0]);
        assert_eq!(tr.into_scalar(), 5.0);

        assert_eq!(bc.dims(), [8, 2, 4]);
        assert_eq!(bc.into_data().to_vec::<f32>().unwrap(), vec![0.0; 64]);
    }
}
