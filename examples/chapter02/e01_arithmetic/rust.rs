use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;
type T1 = Tensor<Backend, 1>;

fn build() -> (T1, T1, T1, T1) {
    let device = Default::default();
    let a = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);
    let b = Tensor::<Backend, 1>::from_floats([10.0, 20.0, 30.0], &device);

    let s = a.clone() + b.clone(); // element-wise add
    let p = a.clone() * b.clone(); // element-wise multiply, NOT matmul
    let sc = a.clone().mul_scalar(2.0);
    let neg = -a;

    (s, p, sc, neg)
}

fn main() {
    let (s, p, sc, neg) = build();
    println!("sum    = {}", s.to_data());
    println!("prod   = {}", p.to_data());
    println!("scaled = {}", sc.to_data());
    println!("neg    = {}", neg.to_data());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (s, p, sc, neg) = build();
        assert_eq!(s.into_data().to_vec::<f32>().unwrap(), vec![11.0, 22.0, 33.0]);
        assert_eq!(p.into_data().to_vec::<f32>().unwrap(), vec![10.0, 40.0, 90.0]);
        assert_eq!(sc.into_data().to_vec::<f32>().unwrap(), vec![2.0, 4.0, 6.0]);
        assert_eq!(neg.into_data().to_vec::<f32>().unwrap(), vec![-1.0, -2.0, -3.0]);
    }
}
