use burn::backend::{Autodiff, NdArray};
use burn::tensor::module::{conv2d, max_pool2d};
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Tensor, TensorData};

type B = Autodiff<NdArray>;

// A tiny LCG so Rust and Python draw exactly the same numbers.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    // A multiple of 1/4 in [-1, 1]. Sums of products of these are exact in f32,
    // so the hand-written and autodiff gradients can agree to the last bit.
    fn quarter(&mut self) -> f32 {
        ((self.next() % 9) as f32 - 4.0) / 4.0
    }
}

const N: usize = 2; // batch
const CI: usize = 2; // input channels
const CO: usize = 3; // output channels
const H: usize = 6; // input height = width
const K: usize = 3; // kernel size
const HO: usize = H - K + 1; // conv output: 4x4 (stride 1, no padding)
const HP: usize = HO / 2; // after 2x2 max-pool: 2x2

// "Valid" cross-correlation of one [ih, iw] map with one [kh, kw] kernel.
fn corr2d(img: &[f32], ih: usize, iw: usize, ker: &[f32], kh: usize, kw: usize) -> Vec<f32> {
    let (oh, ow) = (ih - kh + 1, iw - kw + 1);
    let mut out = vec![0.0f32; oh * ow];
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

// Surround an [h, w] map with p rows and columns of zeros.
fn pad(img: &[f32], h: usize, w: usize, p: usize) -> Vec<f32> {
    let pw = w + 2 * p;
    let mut out = vec![0.0f32; (h + 2 * p) * pw];
    for r in 0..h {
        for c in 0..w {
            out[(r + p) * pw + c + p] = img[r * w + c];
        }
    }
    out
}

// Turn a kernel by 180 degrees: reverse its rows and its columns.
fn rot180(ker: &[f32]) -> Vec<f32> {
    ker.iter().rev().copied().collect()
}

// Slices of the flat [N, C, h, w] buffers.
fn plane(t: &[f32], n: usize, c: usize, channels: usize, side: usize) -> &[f32] {
    let len = side * side;
    &t[(n * channels + c) * len..][..len]
}

struct Forward {
    y: Vec<f32>,        // conv output   [N, CO, 4, 4]
    argmax: Vec<usize>, // for each pooled value, its flat position in y
    loss: f32,
}

fn forward(x: &[f32], w: &[f32], b: &[f32], r: &[f32]) -> Forward {
    let mut y = vec![0.0f32; N * CO * HO * HO];
    for n in 0..N {
        for o in 0..CO {
            let out = &mut y[(n * CO + o) * HO * HO..][..HO * HO];
            out.iter_mut().for_each(|v| *v = b[o]);
            for c in 0..CI {
                let part = corr2d(plane(x, n, c, CI, H), H, H, plane(w, o, c, CI, K), K, K);
                out.iter_mut().zip(part).for_each(|(v, p)| *v += p);
            }
        }
    }
    // 2x2 max-pool, stride 2, remembering where each maximum came from.
    let (mut argmax, mut loss) = (Vec::new(), 0.0f32);
    for n in 0..N {
        for o in 0..CO {
            for i in 0..HP {
                for j in 0..HP {
                    let base = (n * CO + o) * HO * HO;
                    let at = [(0, 0), (0, 1), (1, 0), (1, 1)]
                        .map(|(a, e)| base + (2 * i + a) * HO + 2 * j + e)
                        .into_iter()
                        .reduce(|best, p| if y[p] > y[best] { p } else { best })
                        .unwrap();
                    argmax.push(at);
                    loss += y[at] * r[argmax.len() - 1]; // L = sum(pooled * R)
                }
            }
        }
    }
    Forward { y, argmax, loss }
}

struct Grads {
    gy: Vec<f32>, // dL/dy, the upstream gradient of the conv  [N, CO, 4, 4]
    db: Vec<f32>, // [CO]
    dw: Vec<f32>, // [CO, CI, 3, 3]
    dx: Vec<f32>, // [N, CI, 6, 6]
}

fn backward_by_hand(x: &[f32], w: &[f32], f: &Forward, r: &[f32]) -> Grads {
    // 1. Max-pool: each pooled value's gradient goes to its argmax, the rest get 0.
    let mut gy = vec![0.0f32; N * CO * HO * HO];
    for (k, &at) in f.argmax.iter().enumerate() {
        gy[at] += r[k]; // dL/dpooled = R, because L = sum(pooled * R)
    }
    // 2. Bias: it was added at every position, so its gradient is the sum of gy.
    let mut db = vec![0.0f32; CO];
    for n in 0..N {
        for o in 0..CO {
            db[o] += plane(&gy, n, o, CO, HO).iter().sum::<f32>();
        }
    }
    // 3. Weights: correlate the INPUT with the upstream gradient (gy is the "kernel").
    let mut dw = vec![0.0f32; CO * CI * K * K];
    for o in 0..CO {
        for c in 0..CI {
            for n in 0..N {
                let part = corr2d(
                    plane(x, n, c, CI, H),
                    H,
                    H,
                    plane(&gy, n, o, CO, HO),
                    HO,
                    HO,
                );
                let dst = &mut dw[(o * CI + c) * K * K..][..K * K];
                dst.iter_mut().zip(part).for_each(|(v, p)| *v += p);
            }
        }
    }
    // 4. Input: FULL correlation of gy with the 180-degree-rotated kernel,
    //    i.e. pad gy by K-1 and run an ordinary valid correlation.
    let mut dx = vec![0.0f32; N * CI * H * H];
    for n in 0..N {
        for c in 0..CI {
            for o in 0..CO {
                let padded = pad(plane(&gy, n, o, CO, HO), HO, HO, K - 1); // 8x8
                let flipped = rot180(plane(w, o, c, CI, K));
                let part = corr2d(&padded, HO + 2 * (K - 1), HO + 2 * (K - 1), &flipped, K, K);
                let dst = &mut dx[(n * CI + c) * H * H..][..H * H];
                dst.iter_mut().zip(part).for_each(|(v, p)| *v += p);
            }
        }
    }
    Grads { gy, db, dw, dx }
}

// The same model in Burn, differentiated by autodiff.
fn backward_by_burn(x: &[f32], w: &[f32], b: &[f32], r: &[f32]) -> (f32, Grads) {
    let device = Default::default();
    let t = |v: &[f32], shape: [usize; 4]| {
        Tensor::<B, 4>::from_data(TensorData::new(v.to_vec(), shape), &device)
    };
    let xt = t(x, [N, CI, H, H]).require_grad();
    let wt = t(w, [CO, CI, K, K]).require_grad();
    let bt = Tensor::<B, 1>::from_data(TensorData::new(b.to_vec(), [CO]), &device).require_grad();
    let y = conv2d(
        xt.clone(),
        wt.clone(),
        Some(bt.clone()),
        ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
    );
    let pooled = max_pool2d(y, [2, 2], [2, 2], [0, 0], [1, 1], false);
    let loss = (pooled * t(r, [N, CO, HP, HP])).sum();
    let grads = loss.backward();
    let vec = |g: Tensor<NdArray, 4>| -> Vec<f32> { g.into_data().to_vec().unwrap() };
    let db = bt.grad(&grads).unwrap().into_data().to_vec().unwrap();
    let out = Grads {
        gy: Vec::new(), // autodiff keeps this one to itself
        db,
        dw: vec(wt.grad(&grads).unwrap()),
        dx: vec(xt.grad(&grads).unwrap()),
    };
    (loss.into_scalar(), out)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f32::max)
}

fn has_tie(y: &[f32]) -> bool {
    (0..N * CO * HP * HP).any(|k| {
        let (nc, i, j) = (k / (HP * HP), (k / HP) % HP, k % HP);
        let base = nc * HO * HO;
        let mut v: Vec<f32> = [(0, 0), (0, 1), (1, 0), (1, 1)]
            .iter()
            .map(|&(a, e)| y[base + (2 * i + a) * HO + 2 * j + e])
            .collect();
        v.sort_by(|p, q| q.partial_cmp(p).unwrap());
        v[0] == v[1]
    })
}

fn data() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Lcg(2);
    let x: Vec<f32> = (0..N * CI * H * H).map(|_| rng.quarter()).collect();
    let w: Vec<f32> = (0..CO * CI * K * K).map(|_| rng.quarter()).collect();
    let b = vec![0.5, -0.25, 0.0];
    let r: Vec<f32> = (0..N * CO * HP * HP).map(|_| rng.quarter()).collect();
    (x, w, b, r)
}

fn build() -> (f32, f32, Grads, Grads, bool) {
    let (x, w, b, r) = data();
    let f = forward(&x, &w, &b, &r);
    let tie = has_tie(&f.y);
    let manual = backward_by_hand(&x, &w, &f, &r);
    let (burn_loss, auto) = backward_by_burn(&x, &w, &b, &r);
    (f.loss, burn_loss, manual, auto, tie)
}

fn show(name: &str, v: &[f32], cols: usize) {
    println!("{name} =");
    for row in v.chunks(cols) {
        let line: String = row.iter().map(|x| format!("{x:8.4}")).collect();
        println!("  {line}");
    }
}

fn main() {
    let (x, w, _, r) = data();
    let (loss, burn_loss, manual, auto, tie) = build();
    println!(
        "x {:?}  W {:?}  y {:?}  pooled {:?}",
        [N, CI, H, H],
        [CO, CI, K, K],
        [N, CO, HO, HO],
        [N, CO, HP, HP]
    );
    println!("ties inside a pooling window: {tie}");
    println!("loss: by hand {loss}   burn {burn_loss}");
    show("x[0, 0]", &x[..36], 6);
    show("W[0, 0]", &w[..9], 3);
    show("R[0, 0]   (dL/dpooled)", &r[..4], 2);
    show("gy[0, 0]  (routed to the argmax)", &manual.gy[..16], 4);
    println!("db  by hand {:?}", manual.db);
    println!("db  burn    {:?}", auto.db);
    show("dW[0, 0] by hand", &manual.dw[..9], 3);
    show("dx[0, 0] by hand", &manual.dx[..36], 6);
    println!(
        "max|hand - burn|: db {:?}  dW {:?}  dx {:?}",
        max_diff(&manual.db, &auto.db),
        max_diff(&manual.dw, &auto.dw),
        max_diff(&manual.dx, &auto.dx)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (loss, burn_loss, manual, auto, tie) = build();
        assert!(!tie, "a tie would make the argmax ambiguous");
        assert_eq!(loss, burn_loss);
        // Dyadic data: every sum is exact, so hand and autodiff agree bit for bit.
        assert_eq!(manual.db, auto.db);
        assert_eq!(manual.dw, auto.dw);
        assert_eq!(manual.dx, auto.dx);
        // Numbers from torch .backward() in python/main.py.
        assert_eq!(loss, EXPECTED_LOSS);
        assert_eq!(manual.db, EXPECTED_DB.to_vec());
        assert_eq!(&manual.dw[..9], &EXPECTED_DW00);
        assert_eq!(manual.dx.iter().sum::<f32>(), EXPECTED_DX_SUM);
    }

    const EXPECTED_LOSS: f32 = 0.375;
    const EXPECTED_DB: [f32; 3] = [1.25, -0.25, -0.75];
    #[rustfmt::skip]
    const EXPECTED_DW00: [f32; 9] = [
        -1.25, 0.875, -0.75,
        0.5, -0.9375, -0.9375,
        -0.5625, 1.5625, 0.5625,
    ];
    const EXPECTED_DX_SUM: f32 = -0.1875;
}
