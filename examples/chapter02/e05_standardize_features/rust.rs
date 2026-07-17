use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;
type T2 = Tensor<Backend, 2>;

fn build() -> (T2, T2, T2) {
    let device = Default::default();

    // x: [n_samples, n_features]
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);

    let mean = x.clone().mean_dim(0); // [[3, 4]]
    // var() applies Bessel's correction (unbiased, /(n-1)) -- same as torch's default.
    let std = x.clone().var(0).sqrt(); // [[2, 2]]
    let z = (x - mean.clone()) / std.clone();

    (mean, std, z)
}

fn main() {
    let (mean, std, z) = build();
    println!("mean = {}", mean.to_data());
    println!("std  = {}", std.to_data());
    println!("z    =\n{}", z.to_data());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(got: Vec<f32>, want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn matches_pytorch() {
        let (mean, std, z) = build();
        approx(mean.into_data().to_vec().unwrap(), &[3.0, 4.0]);
        approx(std.into_data().to_vec().unwrap(), &[2.0, 2.0]);
        approx(
            z.into_data().to_vec().unwrap(),
            &[-1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
        );
    }
}
