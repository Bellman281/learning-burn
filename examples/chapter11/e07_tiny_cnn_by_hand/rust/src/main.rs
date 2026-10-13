use burn::tensor::activation::{log_softmax, relu};
use burn::tensor::module::{conv2d, max_pool2d};
use burn::tensor::ops::{ConvOptions, MaxPoolOptions};
use burn::tensor::{Device, Int, Tensor, TensorData};

const SIZE: usize = 12; // image side
const C: usize = 4; // conv filters
const P: usize = SIZE / 2; // side after the 2x2 max-pool
const FEAT: usize = C * P * P; // 144 features into the linear layer
const CLASSES: usize = 4;
const BATCH: usize = 16;
const LR: f32 = 0.3;
const EPOCHS: usize = 10;

// --- the data: 12x12 "shapes", the same generator and LCG as python/main.py ---
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    fn below(&mut self, n: u32) -> usize {
        (self.next() % n) as usize
    }
}

// class 0 = horizontal bar, 1 = vertical bar, 2 = diagonal, 3 = plus sign
fn make_image(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut img = vec![0.0f32; SIZE * SIZE];
    match class {
        0 => {
            let (r, c) = (rng.below(12), rng.below(7));
            (0..6).for_each(|j| img[r * SIZE + c + j] = 1.0);
        }
        1 => {
            let (c, r) = (rng.below(12), rng.below(7));
            (0..6).for_each(|i| img[(r + i) * SIZE + c] = 1.0);
        }
        2 => {
            let (r, c) = (rng.below(7), rng.below(7));
            (0..6).for_each(|i| img[(r + i) * SIZE + c + i] = 1.0);
        }
        _ => {
            let (r, c) = (2 + rng.below(8), 2 + rng.below(8));
            for d in 0..5 {
                img[r * SIZE + c + d - 2] = 1.0;
                img[(r + d - 2) * SIZE + c] = 1.0;
            }
        }
    }
    // background noise in [0, 0.25): exact in f32, so both languages agree bit for bit
    img.iter_mut()
        .for_each(|p| *p += rng.next() as f32 / 32768.0 * 0.25);
    img
}

fn shapes(per_class: usize, seed: u32) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut rng = Lcg(seed);
    let images = (0..per_class * 4)
        .map(|i| make_image(i % 4, &mut rng))
        .collect();
    (images, (0..per_class * 4).map(|i| i % 4).collect())
}

// --- the parameters: four plain Vec<f32>, no Module ---
#[derive(Clone)]
struct Params {
    cw: Vec<f32>, // conv weight   [4, 1, 3, 3]
    cb: Vec<f32>, // conv bias     [4]
    lw: Vec<f32>, // linear weight [144, 4]  (in, out), as in Chapter 8
    lb: Vec<f32>, // linear bias   [4]
}

fn det(n: usize, fan_in: usize, phase: f64) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f64 * 2.3 + phase).sin() / (fan_in as f64).sqrt()) as f32)
        .collect()
}

fn init() -> Params {
    Params {
        cw: det(36, 9, 0.1),
        cb: vec![0.0; C],
        lw: det(FEAT * CLASSES, FEAT, 0.5),
        lb: vec![0.0; CLASSES],
    }
}

// What the forward pass keeps for the backward pass, for ONE image.
struct Cache {
    pre: Vec<f32>,      // conv output before relu  [4, 12, 12]
    argmax: Vec<usize>, // where each pooled value came from, an index into `pre`
    h: Vec<f32>,        // pooled features          [144]
    logits: Vec<f32>,   // [4]
}

// Zero padding: a pixel outside the image reads as 0.
fn pixel(img: &[f32], r: isize, c: isize) -> f32 {
    let inside = r >= 0 && c >= 0 && r < SIZE as isize && c < SIZE as isize;
    if inside {
        img[r as usize * SIZE + c as usize]
    } else {
        0.0
    }
}

fn forward(p: &Params, img: &[f32]) -> Cache {
    // conv 3x3, padding 1: 12x12 -> 12x12
    let mut pre = vec![0.0f32; C * SIZE * SIZE];
    for o in 0..C {
        for i in 0..SIZE {
            for j in 0..SIZE {
                let mut acc = p.cb[o];
                for a in 0..3 {
                    for e in 0..3 {
                        let (r, c) = ((i + a) as isize - 1, (j + e) as isize - 1);
                        acc += p.cw[o * 9 + a * 3 + e] * pixel(img, r, c);
                    }
                }
                pre[(o * SIZE + i) * SIZE + j] = acc;
            }
        }
    }
    // relu, then 2x2 max-pool: 12x12 -> 6x6, remembering the argmax
    let (mut h, mut argmax) = (vec![0.0f32; FEAT], vec![0usize; FEAT]);
    for o in 0..C {
        for i in 0..P {
            for j in 0..P {
                let f = (o * P + i) * P + j;
                let (mut best, mut at) = (f32::NEG_INFINITY, 0);
                for (a, e) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let idx = (o * SIZE + 2 * i + a) * SIZE + 2 * j + e;
                    let v = pre[idx].max(0.0); // relu
                    if v > best {
                        (best, at) = (v, idx);
                    }
                }
                (h[f], argmax[f]) = (best, at);
            }
        }
    }
    // linear: logits = h @ lw + lb
    let logits = (0..CLASSES)
        .map(|k| p.lb[k] + (0..FEAT).map(|f| h[f] * p.lw[f * CLASSES + k]).sum::<f32>())
        .collect();
    Cache {
        pre,
        argmax,
        h,
        logits,
    }
}

// Mean cross-entropy over the batch, and every gradient, by hand.
fn loss_and_grads(p: &Params, images: &[Vec<f32>], labels: &[usize]) -> (f32, Params) {
    let n = images.len() as f32;
    let mut g = Params {
        cw: vec![0.0; 36],
        cb: vec![0.0; C],
        lw: vec![0.0; FEAT * CLASSES],
        lb: vec![0.0; CLASSES],
    };
    let mut loss = 0.0f32;
    for (img, &y) in images.iter().zip(labels) {
        let cache = forward(p, img);
        // softmax cross-entropy (Chapter 6): dL/dlogits = (softmax - onehot) / N
        let m = cache
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = cache.logits.iter().map(|z| (z - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        loss += sum.ln() + m - cache.logits[y];
        let dz: Vec<f32> = (0..CLASSES)
            .map(|k| (exps[k] / sum - if k == y { 1.0 } else { 0.0 }) / n)
            .collect();
        // linear layer (Chapter 8): dW = h^T dz, db = dz, dh = dz W^T
        let mut dh = vec![0.0f32; FEAT];
        for f in 0..FEAT {
            for k in 0..CLASSES {
                g.lw[f * CLASSES + k] += cache.h[f] * dz[k];
                dh[f] += dz[k] * p.lw[f * CLASSES + k];
            }
        }
        (0..CLASSES).for_each(|k| g.lb[k] += dz[k]);
        // max-pool: route each gradient to its argmax; relu: keep it only where pre > 0
        let mut gpre = vec![0.0f32; C * SIZE * SIZE];
        for f in 0..FEAT {
            gpre[cache.argmax[f]] += dh[f];
        }
        for (gv, &pv) in gpre.iter_mut().zip(&cache.pre) {
            if pv <= 0.0 {
                *gv = 0.0;
            }
        }
        // conv (Example 11.5): db = sum of gpre, dW = (padded input) correlated with gpre
        for o in 0..C {
            for i in 0..SIZE {
                for j in 0..SIZE {
                    let gv = gpre[(o * SIZE + i) * SIZE + j];
                    if gv == 0.0 {
                        continue;
                    }
                    g.cb[o] += gv;
                    for a in 0..3 {
                        for e in 0..3 {
                            let (r, c) = ((i + a) as isize - 1, (j + e) as isize - 1);
                            g.cw[o * 9 + a * 3 + e] += gv * pixel(img, r, c);
                        }
                    }
                }
            }
        }
        // (no dL/dx: nothing upstream of the image needs it)
    }
    (loss / n, g)
}

// Plain SGD, written out: param <- param - lr * grad.
fn sgd(p: &mut Params, g: &Params) {
    let pairs = [
        (&mut p.cw, &g.cw),
        (&mut p.cb, &g.cb),
        (&mut p.lw, &g.lw),
        (&mut p.lb, &g.lb),
    ];
    for (param, grad) in pairs {
        param.iter_mut().zip(grad).for_each(|(w, d)| *w -= LR * d);
    }
}

// The check: the same first step in Burn, with autodiff doing the backward pass.
fn burn_first_step(p: &Params, images: &[Vec<f32>], labels: &[usize]) -> (f32, Params) {
    // Only this check uses autodiff. The training loop does not.
    let device = Device::flex().autodiff();
    let n = images.len();
    let pixels: Vec<f32> = images.iter().flatten().copied().collect();
    let x = Tensor::<4>::from_data(TensorData::new(pixels, [n, 1, SIZE, SIZE]), &device);
    let cw =
        Tensor::<4>::from_data(TensorData::new(p.cw.clone(), [C, 1, 3, 3]), &device).require_grad();
    let cb = Tensor::<1>::from_data(TensorData::new(p.cb.clone(), [C]), &device).require_grad();
    let lw = Tensor::<2>::from_data(TensorData::new(p.lw.clone(), [FEAT, CLASSES]), &device)
        .require_grad();
    let lb =
        Tensor::<1>::from_data(TensorData::new(p.lb.clone(), [CLASSES]), &device).require_grad();
    let y: Vec<i64> = labels.iter().map(|&l| l as i64).collect();
    let y = Tensor::<2, Int>::from_data(TensorData::new(y, [n, 1]), &device);

    let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
    let h = relu(conv2d(x, cw.clone(), Some(cb.clone()), options));
    let h = max_pool2d(h, MaxPoolOptions::new([2, 2])).reshape([n, FEAT]);
    let logits = h.matmul(lw.clone()) + lb.clone().unsqueeze::<2>();
    let loss = log_softmax(logits, 1).gather(1, y).mean().neg();
    let grads = loss.backward();
    let v = |t: TensorData| -> Vec<f32> { t.try_to_vec().unwrap() };
    let g = Params {
        cw: v(cw.grad(&grads).unwrap().into_data()),
        cb: v(cb.grad(&grads).unwrap().into_data()),
        lw: v(lw.grad(&grads).unwrap().into_data()),
        lb: v(lb.grad(&grads).unwrap().into_data()),
    };
    (loss.into_scalar(), g)
}

fn max_diff(a: &Params, b: &Params) -> f32 {
    [
        (&a.cw, &b.cw),
        (&a.cb, &b.cb),
        (&a.lw, &b.lw),
        (&a.lb, &b.lb),
    ]
    .iter()
    .flat_map(|(x, y)| x.iter().zip(y.iter()).map(|(p, q)| (p - q).abs()))
    .fold(0.0, f32::max)
}

struct Run {
    first_loss: f32,
    first_grads: Params,
    burn_loss: f32,
    burn_grads: Params,
    history: Vec<f32>,
    correct: usize,
}

fn build() -> Run {
    let (xs, ys) = shapes(100, 42); // 400 training images
    let (xt, yt) = shapes(20, 7); // 80 test images, a different seed
    let mut p = init();

    let (first_loss, first_grads) = loss_and_grads(&p, &xs[..BATCH], &ys[..BATCH]);
    let (burn_loss, burn_grads) = burn_first_step(&p, &xs[..BATCH], &ys[..BATCH]);

    let mut history = Vec::new();
    for _ in 0..EPOCHS {
        let mut total = 0.0;
        for s in (0..xs.len()).step_by(BATCH) {
            let (loss, g) = loss_and_grads(&p, &xs[s..s + BATCH], &ys[s..s + BATCH]);
            sgd(&mut p, &g);
            total += loss;
        }
        history.push(total / (xs.len() / BATCH) as f32);
    }
    let correct = xt
        .iter()
        .zip(&yt)
        .filter(|&(img, &y)| {
            let z = forward(&p, img).logits;
            let best = (0..CLASSES).fold(0, |b, k| if z[k] > z[b] { k } else { b });
            best == y
        })
        .count();
    Run {
        first_loss,
        first_grads,
        burn_loss,
        burn_grads,
        history,
        correct,
    }
}

fn fmt(v: &[f32]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
    format!("[{}]", items.join(", "))
}

fn main() {
    let run = build();
    let g = &run.first_grads;
    println!("step 1: loss {:.6}", run.first_loss);
    println!("  dL/d conv bias       = {}", fmt(&g.cb));
    println!("  dL/d conv w[0, 0, 0] = {}", fmt(&g.cw[..3]));
    println!("  dL/d linear bias     = {}", fmt(&g.lb));
    println!(
        "  burn autodiff: loss {:.6}, max|hand - burn| over all 620 gradients = {:.1e}",
        run.burn_loss,
        max_diff(g, &run.burn_grads)
    );
    for (e, loss) in run.history.iter().enumerate() {
        println!("epoch {}  mean loss {loss:.6}", e + 1);
    }
    println!("test accuracy = {}/80", run.correct);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let run = build();
        // Step 1, by hand, against Burn's autodiff ...
        assert!((run.first_loss - run.burn_loss).abs() < 1e-6);
        assert!(max_diff(&run.first_grads, &run.burn_grads) < 1e-6);
        // ... and against torch .backward() in python/main.py.
        assert!((run.first_loss - 1.3728741).abs() < 1e-6);
        let torch_cb = [-0.016676951, -0.011590492, -0.036864772, -0.015081336];
        let torch_lb = [0.0010417663, -0.0044964300, 0.0057353172, -0.0022806749];
        let torch_cw = [-0.019147957, -0.008102162, -0.002554308];
        let pairs = run
            .first_grads
            .cb
            .iter()
            .zip(torch_cb)
            .chain(run.first_grads.lb.iter().zip(torch_lb))
            .chain(run.first_grads.cw.iter().zip(torch_cw));
        for (a, b) in pairs {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        // The whole run: mean loss per epoch from PyTorch, same data, same order.
        let expected = [
            1.309250, 1.064798, 0.786106, 0.534784, 0.371480, 0.264373, 0.192489, 0.144322,
            0.112260, 0.090219,
        ];
        for (a, b) in run.history.iter().zip(expected) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        assert_eq!(run.correct, 79);
    }
}
