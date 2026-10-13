use burn::tensor::module::{conv2d, unfold4d};
use burn::tensor::ops::{ConvOptions, UnfoldOptions};
use burn::tensor::{Device, Tensor, TensorData};

// A tiny LCG so Rust and Python draw exactly the same numbers.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    fn small(&mut self) -> f32 {
        (self.next() % 7) as f32 - 3.0 // an integer in -3..=3
    }
}

fn out_len(n: usize, k: usize, padding: usize, stride: usize, dilation: usize) -> usize {
    (n + 2 * padding - dilation * (k - 1) - 1) / stride + 1
}

// im2col: one COLUMN per output position, one ROW per (channel, kernel row, kernel col).
// x is one image, [c_in, h, w] flat. Returns the [c_in*k*k, h_out*w_out] matrix, flat.
fn im2col(
    x: &[f32],
    [ci, h, w]: [usize; 3],
    k: usize,
    stride: usize,
    padding: usize,
) -> (Vec<f32>, usize, usize) {
    let (ho, wo) = (
        out_len(h, k, padding, stride, 1),
        out_len(w, k, padding, stride, 1),
    );
    let n_cols = ho * wo;
    let mut cols = vec![0.0f32; ci * k * k * n_cols];
    for c in 0..ci {
        for a in 0..k {
            for e in 0..k {
                let row = (c * k + a) * k + e;
                for i in 0..ho {
                    for j in 0..wo {
                        let r = (i * stride + a) as isize - padding as isize;
                        let q = (j * stride + e) as isize - padding as isize;
                        if r >= 0 && q >= 0 && (r as usize) < h && (q as usize) < w {
                            cols[row * n_cols + i * wo + j] =
                                x[(c * h + r as usize) * w + q as usize];
                        }
                    }
                }
            }
        }
    }
    (cols, ho, wo)
}

// Shapes and work of the im2col form of one conv layer:
// (columns shape, weight-matrix shape, multiply-adds, entries in the column matrix).
fn cost(
    c_in: usize,
    c_out: usize,
    k: usize,
    ho: usize,
    wo: usize,
) -> ([usize; 2], [usize; 2], usize, usize) {
    let (rows, n_cols) = (c_in * k * k, ho * wo);
    (
        [rows, n_cols],
        [c_out, rows],
        c_out * rows * n_cols,
        rows * n_cols,
    )
}

struct Out {
    cols_shape: [usize; 2],
    w_shape: [usize; 2],
    y_shape: [usize; 4],
    first_patch: Vec<f32>,
    cols_diff: f32,
    y_diff: f32,
    y: Vec<f32>,
}

fn build() -> Out {
    let device = Device::flex();
    let mut rng = Lcg(11);
    let x: Vec<f32> = (0..50).map(|_| rng.small()).collect(); // [1, 2, 5, 5], as in Example 11.2
    let w: Vec<f32> = (0..54).map(|_| rng.small()).collect(); // [3, 2, 3, 3]
    let bias = vec![1.0f32, -1.0, 0.0];

    // 1. Unfold the patches by hand: [2*3*3, 5*5] = [18, 25].
    let (cols, ho, wo) = im2col(&x, [2, 5, 5], 3, 1, 1);
    let cols_t = Tensor::<2>::from_data(TensorData::new(cols.clone(), [18, 25]), &device);

    // 2. The kernels are already rows: [3, 2, 3, 3] read as [3, 18] is the same memory.
    let w_mat = Tensor::<2>::from_data(TensorData::new(w.clone(), [3, 18]), &device);
    let b = Tensor::<2>::from_data(TensorData::new(bias.clone(), [3, 1]), &device);

    // 3. The whole convolution: ONE matmul, plus the bias broadcast along each row.
    let y = (w_mat.clone().matmul(cols_t.clone()) + b).reshape([1, 3, ho, wo]); // [3,18]@[18,25]

    // The checks: Burn's unfold4d and Burn's conv2d.
    let xt = Tensor::<4>::from_data(TensorData::new(x, [1, 2, 5, 5]), &device);
    let unfolded = unfold4d(
        xt.clone(),
        [3, 3],
        UnfoldOptions::new([1, 1], [1, 1], [1, 1]),
    );
    let wt = Tensor::<4>::from_data(TensorData::new(w, [3, 2, 3, 3]), &device);
    let bt = Tensor::<1>::from_data(TensorData::new(bias, [3]), &device);
    let conv = conv2d(
        xt,
        wt,
        Some(bt),
        ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
    );

    let max_abs = |t: Tensor<2>| t.abs().max().into_scalar();
    let cols_diff = max_abs(unfolded.reshape([18, 25]) - cols_t.clone());
    let y_diff = max_abs((y.clone() - conv).reshape([3, 25]));
    Out {
        cols_shape: [18, 25],
        w_shape: w_mat.dims(),
        y_shape: y.dims(),
        first_patch: (0..18).map(|r| cols[r * 25]).collect(),
        cols_diff,
        y_diff,
        y: y.into_data().try_to_vec().unwrap(),
    }
}

fn main() {
    let o = build();
    println!(
        "columns  {:?}   weights {:?}   y {:?}",
        o.cols_shape, o.w_shape, o.y_shape
    );
    println!("first column (the padded patch under output (0, 0)):");
    println!("   channel 0: {:?}", &o.first_patch[..9]);
    println!("   channel 1: {:?}", &o.first_patch[9..]);
    println!("max|im2col - unfold4d|  = {:?}", o.cols_diff);
    println!("max|matmul - conv2d|    = {:?}", o.y_diff);
    println!("y[0, 0] (padding 1) =");
    for r in 0..5 {
        let line: String = (0..5).map(|c| format!("{:6.1}", o.y[r * 5 + c])).collect();
        println!("   {line}");
    }
    for (name, (cs, ws, macs, entries)) in [
        ("this example", cost(2, 3, 3, 5, 5)),
        ("64->64, 3x3, 56x56", cost(64, 64, 3, 56, 56)),
    ] {
        println!(
            "{name:<20} columns {cs:?}  weights {ws:?}  multiply-adds {macs}  column entries {entries}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let o = build();
        assert_eq!(
            (o.cols_shape, o.w_shape, o.y_shape),
            ([18, 25], [3, 18], [1, 3, 5, 5])
        );
        // Same first column as F.unfold in python/main.py.
        #[rustfmt::skip]
        let first = vec![0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -3.0, -1.0,
                         0.0, 0.0, 0.0, 0.0, -1.0, -3.0, 0.0, 3.0, 2.0];
        assert_eq!(o.first_patch, first);
        assert_eq!((o.cols_diff, o.y_diff), (0.0, 0.0));
        // The inner 3x3 of channel 0 is Example 11.2's stride-1 output.
        let inner: Vec<f32> = (1..4)
            .flat_map(|r| (1..4).map(move |c| (r, c)))
            .map(|(r, c)| o.y[r * 5 + c])
            .collect();
        assert_eq!(
            inner,
            vec![-1.0, -5.0, 5.0, 1.0, -35.0, 20.0, 10.0, -12.0, 20.0]
        );
        assert_eq!(cost(2, 3, 3, 5, 5), ([18, 25], [3, 18], 1350, 450));
        assert_eq!(
            cost(64, 64, 3, 56, 56),
            ([576, 3136], [64, 576], 115605504, 1806336)
        );
    }
}
