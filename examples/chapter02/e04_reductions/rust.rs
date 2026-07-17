use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};

type Backend = NdArray;

fn build() -> (
    Tensor<Backend, 1>,
    Tensor<Backend, 2>,
    Tensor<Backend, 2>,
    Tensor<Backend, 2>,
    Tensor<Backend, 2, Int>,
) {
    let device = Default::default();
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

    let total = x.clone().sum(); // scalar 10
    let col_sum = x.clone().sum_dim(0); // [[4, 6]]
    let row_mean = x.clone().mean_dim(1); // [[1.5], [3.5]]
    let (m, idx) = x.max_dim_with_indices(1); // values [[2],[4]], indices [[1],[1]]

    (total, col_sum, row_mean, m, idx)
}

fn main() {
    let (total, col_sum, row_mean, m, idx) = build();
    println!("total     = {}", total.to_data());
    println!("col_sum   = {}", col_sum.to_data());
    println!("row_mean  = {}", row_mean.to_data());
    println!("max vals  = {}", m.to_data());
    println!("max idx   = {}", idx.to_data());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (total, col_sum, row_mean, m, idx) = build();

        assert_eq!(total.into_scalar(), 10.0);
        assert_eq!(col_sum.into_data().to_vec::<f32>().unwrap(), vec![4.0, 6.0]);
        assert_eq!(row_mean.into_data().to_vec::<f32>().unwrap(), vec![1.5, 3.5]);

        // Burn keeps the reduced dim ([2,1]); torch's max(dim) drops it ([2]).
        // Values and indices are identical once flattened.
        assert_eq!(m.into_data().to_vec::<f32>().unwrap(), vec![2.0, 4.0]);
        assert_eq!(idx.into_data().to_vec::<i64>().unwrap(), vec![1, 1]);
    }
}
