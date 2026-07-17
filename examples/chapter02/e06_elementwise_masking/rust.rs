use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;
type T1 = Tensor<Backend, 1>;

fn build() -> (T1, T1, T1) {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([-1.0, 0.0, 2.0], &device);

    let y = x.clone().exp().log1p(); // softplus: log1p(exp(x))
    let c = x.clone().clamp(0.0, 1.0);
    let mask = x.clone().greater_elem(0.0); // Bool: [false, false, true]

    // mask_where puts `value` where the mask is true, self where false
    // == torch.where(mask, value, self)
    let picked = x.mask_where(mask, Tensor::<Backend, 1>::from_floats([9.0, 9.0, 9.0], &device));

    (y, c, picked)
}

fn main() {
    let (y, c, picked) = build();
    println!("softplus = {}", y.to_data());
    println!("clamped  = {}", c.to_data());
    println!("picked   = {}", picked.to_data());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (y, c, picked) = build();

        // softplus, compared with a tolerance
        let yv: Vec<f32> = y.into_data().to_vec().unwrap();
        let expected = [0.313_261_66_f32, 0.693_147_18, 2.126_928_1];
        for i in 0..3 {
            assert!((yv[i] - expected[i]).abs() < 1e-4, "y[{i}] = {}", yv[i]);
        }

        assert_eq!(c.into_data().to_vec::<f32>().unwrap(), vec![0.0, 0.0, 1.0]);
        assert_eq!(
            picked.into_data().to_vec::<f32>().unwrap(),
            vec![-1.0, 0.0, 9.0]
        );
    }
}
