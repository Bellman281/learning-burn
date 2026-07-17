use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};

type Backend = NdArray;
type I1 = Tensor<Backend, 1, Int>;
type I2 = Tensor<Backend, 2, Int>;

fn build() -> (I2, I1, I2, I2, I2) {
    let device = Default::default();

    let t = Tensor::<Backend, 1, Int>::arange(0..12, &device);
    let m = t.reshape([3, 4]);
    let flat = m.clone().flatten::<1>(0, 1); // back to a flat [12]
    let piece = m.clone().slice([0..2, 1..3]); // rows 0-1, cols 1-2
    let col = m.clone().narrow(1, 0, 1); // first column
    let rows = m.clone().select(0, Tensor::<Backend, 1, Int>::from_data([0, 2], &device));

    (m, flat, piece, col, rows)
}

fn main() {
    let (m, flat, piece, col, rows) = build();
    println!("reshaped =\n{}", m);
    println!("flat     =\n{}", flat);
    println!("piece    =\n{}", piece);
    println!("first col=\n{}", col);
    println!("rows 0,2 =\n{}", rows);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (m, flat, piece, col, rows) = build();

        assert_eq!(m.dims(), [3, 4]);
        assert_eq!(
            m.into_data().to_vec::<i64>().unwrap(),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        );

        assert_eq!(flat.dims(), [12]);
        assert_eq!(
            flat.into_data().to_vec::<i64>().unwrap(),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        );

        assert_eq!(piece.dims(), [2, 2]);
        assert_eq!(piece.into_data().to_vec::<i64>().unwrap(), vec![1, 2, 5, 6]);

        assert_eq!(col.dims(), [3, 1]);
        assert_eq!(col.into_data().to_vec::<i64>().unwrap(), vec![0, 4, 8]);

        assert_eq!(rows.dims(), [2, 4]);
        assert_eq!(
            rows.into_data().to_vec::<i64>().unwrap(),
            vec![0, 1, 2, 3, 8, 9, 10, 11]
        );
    }
}
