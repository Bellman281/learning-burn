use burn::backend::{Autodiff, NdArray};
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Tensor, TensorData};

// A gradient check is run in f64 (Chapter 9), and so is Burn's check here.
type B = Autodiff<NdArray<f64>>;

// A tiny LCG so Rust and Python draw exactly the same numbers.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    fn uniform(&mut self) -> f64 {
        self.next() as f64 / 32768.0 * 2.0 - 1.0 // in [-1, 1), exact in f64
    }
}

const N: usize = 1;
const CI: usize = 2;
const CO: usize = 2;
const H: usize = 5;
const K: usize = 3;
const P: usize = 1; // "same" padding: the output is 5x5 too
const EPS: f64 = 1e-5;

// "Valid" cross-correlation of one [ih, iw] map with one [kh, kw] kernel.
fn corr2d(img: &[f64], ih: usize, iw: usize, ker: &[f64], kh: usize, kw: usize) -> Vec<f64> {
    let (oh, ow) = (ih - kh + 1, iw - kw + 1);
    let mut out = vec![0.0; oh * ow];
    for i in 0..oh {
        for j in 0..ow {
            for a in 0..kh {
                for e in 0..kw {
                    out[i * ow + j] += ker[a * kw + e] * img[(i + a) * iw + j + e];
                }
            }
        }
    }
    out
}

fn pad(img: &[f64], h: usize, w: usize, p: usize) -> Vec<f64> {
    let pw = w + 2 * p;
    let mut out = vec![0.0; (h + 2 * p) * pw];
    for r in 0..h {
        for c in 0..w {
            out[(r + p) * pw + c + p] = img[r * w + c];
        }
    }
    out
}

fn plane(t: &[f64], n: usize, c: usize, channels: usize, side: usize) -> &[f64] {
    &t[(n * channels + c) * side * side..][..side * side]
}

fn conv(x: &[f64], w: &[f64], b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; N * CO * H * H];
    for n in 0..N {
        for o in 0..CO {
            for c in 0..CI {
                let xp = pad(plane(x, n, c, CI, H), H, H, P);
                let part = corr2d(&xp, H + 2 * P, H + 2 * P, plane(w, o, c, CI, K), K, K);
                let dst = &mut y[(n * CO + o) * H * H..][..H * H];
                dst.iter_mut().zip(part).for_each(|(v, p)| *v += p);
            }
            y[(n * CO + o) * H * H..][..H * H]
                .iter_mut()
                .for_each(|v| *v += b[o]);
        }
    }
    y
}

// L = sum(R * tanh(y)): smooth, so finite differences have something to measure.
fn loss_at(x: &[f64], w: &[f64], b: &[f64], r: &[f64]) -> f64 {
    conv(x, w, b).iter().zip(r).map(|(y, r)| r * y.tanh()).sum()
}

// Example 11.5's three formulas, now with padding P. Returns (db, dW, dx).
fn by_hand(x: &[f64], w: &[f64], b: &[f64], r: &[f64], rotate: bool) -> [Vec<f64>; 3] {
    let y = conv(x, w, b);
    let gy: Vec<f64> = y
        .iter()
        .zip(r)
        .map(|(y, r)| r * (1.0 - y.tanh().powi(2)))
        .collect();
    let mut db = vec![0.0; CO];
    let mut dw = vec![0.0; CO * CI * K * K];
    let mut dx = vec![0.0; N * CI * H * H];
    for n in 0..N {
        for o in 0..CO {
            let g = plane(&gy, n, o, CO, H);
            db[o] += g.iter().sum::<f64>();
            for c in 0..CI {
                // dW: correlate the PADDED input with gy.
                let xp = pad(plane(x, n, c, CI, H), H, H, P);
                let part = corr2d(&xp, H + 2 * P, H + 2 * P, g, H, H);
                let dst = &mut dw[(o * CI + c) * K * K..][..K * K];
                dst.iter_mut().zip(part).for_each(|(v, p)| *v += p);
                // dx: pad gy by K-1-P (the full correlation, cropped by the forward
                // padding) and correlate with the 180-degree-rotated kernel.
                let ker = plane(w, o, c, CI, K);
                let ker: Vec<f64> = if rotate {
                    ker.iter().rev().copied().collect()
                } else {
                    ker.to_vec() // the bug: forget to rotate
                };
                let q = K - 1 - P;
                let gp = pad(g, H, H, q);
                let part = corr2d(&gp, H + 2 * q, H + 2 * q, &ker, K, K);
                let dst = &mut dx[(n * CI + c) * H * H..][..H * H];
                dst.iter_mut().zip(part).for_each(|(v, p)| *v += p);
            }
        }
    }
    [db, dw, dx]
}

// Central differences, one entry at a time: two forward passes per entry.
fn numerical(x: &[f64], w: &[f64], b: &[f64], r: &[f64]) -> [Vec<f64>; 3] {
    let params = [b.to_vec(), w.to_vec(), x.to_vec()];
    let mut out: [Vec<f64>; 3] = Default::default();
    for which in 0..3 {
        for i in 0..params[which].len() {
            let (mut plus, mut minus) = (params.clone(), params.clone());
            plus[which][i] += EPS;
            minus[which][i] -= EPS;
            let lp = loss_at(&plus[2], &plus[1], &plus[0], r);
            let lm = loss_at(&minus[2], &minus[1], &minus[0], r);
            out[which].push((lp - lm) / (2.0 * EPS));
        }
    }
    out
}

// The same loss in Burn, differentiated by autodiff (in f64).
fn by_burn(x: &[f64], w: &[f64], b: &[f64], r: &[f64]) -> [Vec<f64>; 3] {
    let device = Default::default();
    let t = |v: &[f64], s: [usize; 4]| {
        Tensor::<B, 4>::from_data(TensorData::new(v.to_vec(), s), &device)
    };
    let xt = t(x, [N, CI, H, H]).require_grad();
    let wt = t(w, [CO, CI, K, K]).require_grad();
    let bt = Tensor::<B, 1>::from_data(TensorData::new(b.to_vec(), [CO]), &device).require_grad();
    let options = ConvOptions::new([1, 1], [P, P], [1, 1], 1);
    let y = conv2d(xt.clone(), wt.clone(), Some(bt.clone()), options);
    let grads = (t(r, [N, CO, H, H]) * y.tanh()).sum().backward();
    [
        bt.grad(&grads).unwrap().into_data().to_vec().unwrap(),
        wt.grad(&grads).unwrap().into_data().to_vec().unwrap(),
        xt.grad(&grads).unwrap().into_data().to_vec().unwrap(),
    ]
}

// Chapter 9's relative error, the worst entry.
fn rel_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1e-12))
        .fold(0.0, f64::max)
}

fn data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Lcg(9);
    let x: Vec<f64> = (0..N * CI * H * H).map(|_| rng.uniform()).collect();
    let w: Vec<f64> = (0..CO * CI * K * K).map(|_| rng.uniform()).collect();
    let b: Vec<f64> = (0..CO).map(|_| rng.uniform()).collect();
    let r: Vec<f64> = (0..N * CO * H * H).map(|_| rng.uniform()).collect();
    (x, w, b, r)
}

struct Report {
    loss: f64,
    hand: [Vec<f64>; 3],
    num: [Vec<f64>; 3],
    burn: [Vec<f64>; 3],
    errs: [f64; 3],
    burn_errs: [f64; 3],
    bug_err: f64,
}

fn build() -> Report {
    let (x, w, b, r) = data();
    let hand = by_hand(&x, &w, &b, &r, true);
    let num = numerical(&x, &w, &b, &r);
    let burn = by_burn(&x, &w, &b, &r);
    let buggy = by_hand(&x, &w, &b, &r, false);
    let errs = [0, 1, 2].map(|i| rel_err(&hand[i], &num[i]));
    let burn_errs = [0, 1, 2].map(|i| rel_err(&hand[i], &burn[i]));
    let bug_err = rel_err(&buggy[2], &num[2]);
    Report {
        loss: loss_at(&x, &w, &b, &r),
        hand,
        num,
        burn,
        errs,
        burn_errs,
        bug_err,
    }
}

fn main() {
    let rep = build();
    let entries: usize = rep.hand.iter().map(|g| g.len()).sum();
    println!("loss = {:.12}", rep.loss);
    println!("checked {entries} gradient entries, eps = {EPS:e}");
    for (name, i) in [
        ("dL/db[0]       ", 0),
        ("dL/dW[0,0,0,0] ", 1),
        ("dL/dx[0,0,0,0] ", 2),
    ] {
        println!(
            "{name} hand {:.12}  burn {:.12}  numerical {:.12}",
            rep.hand[i][0], rep.burn[i][0], rep.num[i][0]
        );
    }
    for (i, name) in ["db", "dW", "dx"].iter().enumerate() {
        println!(
            "max relative error {name}: vs finite differences {:.1e}, vs burn autodiff {:.1e}",
            rep.errs[i], rep.burn_errs[i]
        );
    }
    println!(
        "kernel NOT rotated: max relative error dx = {:.2}",
        rep.bug_err
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let rep = build();
        assert!(rep.errs.iter().all(|&e| e < 1e-7), "{:?}", rep.errs);
        assert!(
            rep.burn_errs.iter().all(|&e| e < 1e-12),
            "{:?}",
            rep.burn_errs
        );
        assert!(rep.bug_err > 0.1);
        // The same three entries from torch autograd in python/main.py.
        let firsts = [rep.burn[0][0], rep.burn[1][0], rep.burn[2][0]];
        let torch = [-5.416377452111, 0.262925182681, -0.150826076058];
        for (a, b) in firsts.iter().zip(torch) {
            assert!((a - b).abs() < 1e-11, "{a} vs {b}");
        }
        assert!((rep.loss - -2.542353235219).abs() < 1e-11);
    }
}
