use burn::tensor::module::{avg_pool2d, max_pool2d_with_indices};
use burn::tensor::ops::{AvgPoolOptions, MaxPoolOptions};
use burn::tensor::{Device, Tensor, TensorData};

// A tiny LCG so Rust and Python draw exactly the same numbers.
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

// 0..n-1 in a random order (Fisher-Yates), so no two pixels tie.
fn shuffled(n: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n).map(|i| i as f32).collect();
    for i in (1..n).rev() {
        let j = rng.below(i as u32 + 1);
        v.swap(i, j);
    }
    v
}

// Max pooling on one [h, w] map, window k, stride s, no padding.
// Returns the maxima and, for each window, WHERE the maximum was: the flat
// index r * w + q into the input. The backward pass will need exactly that.
fn max_pool(x: &[f32], h: usize, w: usize, k: usize, s: usize) -> (Vec<f32>, Vec<usize>) {
    let (ho, wo) = ((h - k) / s + 1, (w - k) / s + 1);
    let (mut vals, mut idx) = (Vec::new(), Vec::new());
    for i in 0..ho {
        for j in 0..wo {
            let (mut best, mut at) = (f32::NEG_INFINITY, 0);
            for a in 0..k {
                for e in 0..k {
                    let (r, q) = (i * s + a, j * s + e);
                    if x[r * w + q] > best {
                        (best, at) = (x[r * w + q], r * w + q);
                    }
                }
            }
            vals.push(best);
            idx.push(at);
        }
    }
    (vals, idx)
}

fn avg_pool(x: &[f32], h: usize, w: usize, k: usize, s: usize) -> Vec<f32> {
    let (ho, wo) = ((h - k) / s + 1, (w - k) / s + 1);
    let mut out = Vec::new();
    for i in 0..ho {
        for j in 0..wo {
            let sum: f32 = (0..k)
                .flat_map(|a| (0..k).map(move |e| (a, e)))
                .map(|(a, e)| x[(i * s + a) * w + j * s + e])
                .sum();
            out.push(sum / (k * k) as f32);
        }
    }
    out
}

// How much of the input one output value can "see", layer by layer:
//   rf grows by (k - 1) * jump; jump (input pixels between neighbouring outputs) *= stride.
fn receptive_fields(layers: &[(&str, usize, usize)]) -> Vec<(String, usize, usize)> {
    let (mut rf, mut jump) = (1, 1);
    layers
        .iter()
        .map(|&(name, k, stride)| {
            rf += (k - 1) * jump;
            jump *= stride;
            (name.to_string(), rf, jump)
        })
        .collect()
}

struct MaxRun {
    name: &'static str,
    vals: Vec<f32>,
    idx: Vec<usize>,
    vals_ok: bool,
    idx_ok: bool,
}

fn build() -> (
    Vec<f32>,
    Vec<MaxRun>,
    Vec<f32>,
    bool,
    Vec<(String, usize, usize)>,
) {
    let device = Device::flex();
    let x = shuffled(36, &mut Lcg(10));
    let xt = Tensor::<4>::from_data(TensorData::new(x.clone(), [1, 1, 6, 6]), &device);

    let mut runs = Vec::new();
    for (name, k, s) in [("max 2x2/2", 2, 2), ("max 3x3/2", 3, 2)] {
        let (vals, idx) = max_pool(&x, 6, 6, k, s);
        // Burn's functional op returns the same two things: values and flat indices.
        let (bv, bi) =
            max_pool2d_with_indices(xt.clone(), MaxPoolOptions::new([k, k]).with_stride([s, s]));
        let bv: Vec<f32> = bv.into_data().try_to_vec().unwrap();
        let bi: Vec<i64> = bi.into_data().convert::<i64>().try_to_vec().unwrap();
        let vals_ok = bv == vals;
        let idx_ok = bi.iter().map(|&i| i as usize).eq(idx.iter().copied());
        runs.push(MaxRun {
            name,
            vals,
            idx,
            vals_ok,
            idx_ok,
        });
    }
    let avg = avg_pool(&x, 6, 6, 2, 2);
    let burn_avg: Vec<f32> = avg_pool2d(xt, AvgPoolOptions::new([2, 2]))
        .into_data()
        .try_to_vec()
        .unwrap();
    let avg_ok = burn_avg == avg;
    let rf = receptive_fields(&[("conv 3x3", 3, 1), ("pool 2x2", 2, 2), ("conv 3x3", 3, 1)]);
    (x, runs, avg, avg_ok, rf)
}

fn main() {
    let (x, runs, avg, avg_ok, rf) = build();
    println!("x =");
    for r in 0..6 {
        let line: String = (0..6).map(|c| format!("{:5.0}", x[r * 6 + c])).collect();
        println!("   {line}");
    }
    for m in &runs {
        let at: Vec<String> = m
            .idx
            .iter()
            .map(|i| format!("({},{})", i / 6, i % 6))
            .collect();
        println!("{}: values {:?}", m.name, m.vals);
        println!("           argmax {:?} = {}", m.idx, at.join(" "));
        println!(
            "           same as burn: values {}, indices {}",
            m.vals_ok, m.idx_ok
        );
    }
    println!("avg 2x2/2: {avg:?}  same as burn: {avg_ok}");
    for (name, size, jump) in rf {
        println!("after {name}: receptive field {size}x{size}, jump {jump}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (_, runs, avg, avg_ok, rf) = build();
        assert!(runs.iter().all(|m| m.vals_ok && m.idx_ok));
        // The flat indices F.max_pool2d(return_indices=True) gives in python/main.py.
        assert_eq!(runs[0].idx, vec![1, 2, 5, 12, 14, 22, 24, 26, 29]);
        assert_eq!(runs[1].idx, vec![14, 14, 14, 14]);
        assert!(avg_ok);
        assert_eq!(
            avg,
            vec![12.75, 11.0, 20.25, 21.75, 22.75, 18.5, 14.75, 22.25, 13.5]
        );
        let sizes: Vec<usize> = rf.iter().map(|r| r.1).collect();
        assert_eq!(sizes, vec![3, 4, 8]);
    }
}
