use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
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

// The output-size rule, one spatial axis at a time.
fn out_len(n: usize, k: usize, padding: usize, stride: usize, dilation: usize) -> usize {
    (n + 2 * padding - dilation * (k - 1) - 1) / stride + 1
}

// Cross-correlation, the definition written out as loops.
// x is [n, c_in, h, w] and w is [c_out, c_in, kh, kw], both flat and row-major.
#[allow(clippy::too_many_arguments)]
fn conv2d_loops(
    x: &[f32],
    [n, ci, h, wd]: [usize; 4],
    w: &[f32],
    [co, _, kh, kw]: [usize; 4],
    bias: &[f32],
    stride: usize,
    padding: usize,
    dilation: usize,
) -> (Vec<f32>, [usize; 4]) {
    let ho = out_len(h, kh, padding, stride, dilation);
    let wo = out_len(wd, kw, padding, stride, dilation);
    let mut y = vec![0.0f32; n * co * ho * wo];
    for b in 0..n {
        for o in 0..co {
            for i in 0..ho {
                for j in 0..wo {
                    let mut acc = bias[o];
                    for c in 0..ci {
                        for a in 0..kh {
                            for e in 0..kw {
                                // Where tap (a, e) lands in the unpadded input.
                                let r = (i * stride + a * dilation) as isize - padding as isize;
                                let q = (j * stride + e * dilation) as isize - padding as isize;
                                if r < 0 || q < 0 || r >= h as isize || q >= wd as isize {
                                    continue; // a padding zero contributes nothing
                                }
                                let (r, q) = (r as usize, q as usize);
                                acc += w[((o * ci + c) * kh + a) * kw + e]
                                    * x[((b * ci + c) * h + r) * wd + q];
                            }
                        }
                    }
                    y[((b * co + o) * ho + i) * wo + j] = acc;
                }
            }
        }
    }
    (y, [n, co, ho, wo])
}

type Case = (&'static str, usize, usize, usize); // (name, stride, padding, dilation)
const CASES: [Case; 4] = [
    ("stride 1", 1, 0, 1),
    ("padding 1", 1, 1, 1),
    ("stride 2, padding 1", 2, 1, 1),
    ("dilation 2, padding 2", 1, 2, 2),
];

struct Run {
    name: &'static str,
    rule: usize,
    loops_shape: [usize; 4],
    burn_shape: [usize; 4],
    max_diff: f32,
    y: Vec<f32>,
}

/// Returns (x, w, one Run per case, parameter count).
fn build() -> (Vec<f32>, Vec<f32>, Vec<Run>, usize) {
    let device = Device::flex();
    let mut rng = Lcg(11);
    let (x_shape, w_shape) = ([1, 2, 5, 5], [3, 2, 3, 3]);
    let x: Vec<f32> = (0..50).map(|_| rng.small()).collect();
    let w: Vec<f32> = (0..54).map(|_| rng.small()).collect();
    let bias = vec![1.0f32, -1.0, 0.0];

    let xt = Tensor::<4>::from_data(TensorData::new(x.clone(), x_shape), &device);
    let wt = Tensor::<4>::from_data(TensorData::new(w.clone(), w_shape), &device);
    let bt = Tensor::<1>::from_data(TensorData::new(bias.clone(), [3]), &device);

    let mut runs = Vec::new();
    for (name, s, p, d) in CASES {
        let (y, loops_shape) = conv2d_loops(&x, x_shape, &w, w_shape, &bias, s, p, d);
        let options = ConvOptions::new([s, s], [p, p], [d, d], 1);
        let yt = conv2d(xt.clone(), wt.clone(), Some(bt.clone()), options);
        let burn_shape = yt.dims();
        let burn: Vec<f32> = yt.into_data().try_to_vec().unwrap();
        let max_diff = y
            .iter()
            .zip(&burn)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let rule = out_len(5, 3, p, s, d);
        runs.push(Run {
            name,
            rule,
            loops_shape,
            burn_shape,
            max_diff,
            y,
        });
    }
    // C_out * C_in * kh * kw weights, plus one bias per output channel.
    let params = w_shape.iter().product::<usize>() + w_shape[0];
    (x, w, runs, params)
}

fn grid(v: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let line: String = (0..cols)
            .map(|c| format!("{:6.1}", v[r * cols + c]))
            .collect();
        println!("   {line}");
    }
}

fn main() {
    let (x, w, runs, params) = build();
    for c in 0..2 {
        println!("x[0, {c}] =");
        grid(&x[c * 25..], 5, 5);
    }
    for c in 0..2 {
        println!("w[0, {c}] =");
        grid(&w[c * 9..], 3, 3);
    }
    println!("bias    = [1.0, -1.0, 0.0]");
    println!("params  = {params}  (3*2*3*3 weights + 3 biases)");
    for r in &runs {
        println!(
            "{:<22} rule {}x{}  loops {:?}  burn {:?}  max|loops-burn| = {:?}",
            r.name, r.rule, r.rule, r.loops_shape, r.burn_shape, r.max_diff
        );
    }
    for o in 0..3 {
        println!("stride 1: y[0, {o}] =");
        grid(&runs[0].y[o * 9..], 3, 3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (_, _, runs, params) = build();
        assert_eq!(params, 57);
        let rules: Vec<usize> = runs.iter().map(|r| r.rule).collect();
        assert_eq!(rules, vec![3, 5, 3, 5]);
        assert!(runs.iter().all(|r| r.loops_shape == r.burn_shape));
        assert!(runs.iter().all(|r| r.max_diff == 0.0));
        // The stride-1 output from F.conv2d in python/main.py, channel by channel.
        #[rustfmt::skip]
        let expected = vec![
            -1.0, -5.0, 5.0, 1.0, -35.0, 20.0, 10.0, -12.0, 20.0,
            -10.0, 7.0, 4.0, 6.0, 3.0, -10.0, -15.0, -3.0, 19.0,
            13.0, 9.0, -17.0, -11.0, 2.0, -38.0, 1.0, 32.0, -18.0,
        ];
        assert_eq!(runs[0].y, expected);
    }
}
