use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::{Module, Param};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput, TransformerEncoderLayer,
};
use burn::nn::{LayerNorm, Linear};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::{Bool, Tensor, TensorData};

// No autodiff: every Dropout inside the encoder is switched off (Example 13.5).
type B = NdArray;

const D_MODEL: usize = 8;
const N_HEADS: usize = 2;
const D_FF: usize = 16;
const N_LAYERS: usize = 2;
const SEQ: usize = 5;

// --- fixed weights: every tensor gets its own phase, in the same order as main.py ---
struct Filler {
    phase: f64,
}
impl Filler {
    fn wave(&mut self, n: usize, step: f64, scale: f64, offset: f64) -> Vec<f32> {
        let p = self.phase;
        self.phase += 1.0;
        (0..n)
            .map(|i| (offset + scale * (i as f64 * step + p).sin()) as f32)
            .collect()
    }
    // A PyTorch-layout Linear weight [out, in], handed to Burn as [in, out].
    fn linear(&mut self, lin: &mut Linear<B>, d_in: usize, d_out: usize, device: &NdArrayDevice) {
        let w = self.wave(d_out * d_in, 2.3, 1.0 / (d_in as f64).sqrt(), 0.0);
        let w = Tensor::<B, 2>::from_data(TensorData::new(w, [d_out, d_in]), device);
        lin.weight = Param::from_tensor(w.transpose());
        let b = self.wave(d_out, 1.7, 0.1, 0.0);
        lin.bias = Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(b, [d_out]),
            device,
        )));
    }
    fn norm(&mut self, ln: &mut LayerNorm<B>, device: &NdArrayDevice) {
        let g = self.wave(D_MODEL, 1.7, 0.1, 1.0);
        let b = self.wave(D_MODEL, 1.7, 0.1, 0.0);
        ln.gamma = Param::from_tensor(Tensor::from_data(TensorData::new(g, [D_MODEL]), device));
        ln.beta = Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(b, [D_MODEL]),
            device,
        )));
    }
}

fn encoder(device: &NdArrayDevice) -> TransformerEncoder<B> {
    // Post-norm (norm_first = false), GELU, dropout 0.1: all Burn's defaults.
    let mut enc = TransformerEncoderConfig::new(D_MODEL, D_FF, N_HEADS, N_LAYERS).init::<B>(device);
    let mut f = Filler { phase: 0.0 };
    for layer in enc.layers.iter_mut() {
        f.linear(&mut layer.mha.query, D_MODEL, D_MODEL, device);
        f.linear(&mut layer.mha.key, D_MODEL, D_MODEL, device);
        f.linear(&mut layer.mha.value, D_MODEL, D_MODEL, device);
        f.linear(&mut layer.mha.output, D_MODEL, D_MODEL, device);
        f.linear(&mut layer.pwff.linear_inner, D_MODEL, D_FF, device);
        f.linear(&mut layer.pwff.linear_outer, D_FF, D_MODEL, device);
        f.norm(&mut layer.norm_1, device);
        f.norm(&mut layer.norm_2, device);
    }
    enc
}

fn input(device: &NdArrayDevice) -> Tensor<B, 3> {
    let v: Vec<f32> = (0..2 * SEQ * D_MODEL)
        .map(|i| (i as f32 * 0.37).sin())
        .collect();
    Tensor::from_data(TensorData::new(v, [2, SEQ, D_MODEL]), device)
}

// --- Chapter 10's block, written out again, reading the weights from Burn's layer ---
fn layer_norm(x: Tensor<B, 3>, ln: &LayerNorm<B>) -> Tensor<B, 3> {
    let mean = x.clone().mean_dim(2);
    let centered = x - mean;
    let var = centered.clone().square().mean_dim(2); // biased: divide by d
    let gamma = ln.gamma.val().reshape([1, 1, D_MODEL]);
    let beta = ln.beta.as_ref().unwrap().val().reshape([1, 1, D_MODEL]);
    centered / var.add_scalar(1e-5).sqrt() * gamma + beta
}

fn layer_by_hand(layer: &TransformerEncoderLayer<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, seq, _] = x.dims();
    let d_k = D_MODEL / N_HEADS;
    // [batch, seq, d_model] -> [batch, heads, seq, d_k]: head h owns features h*d_k..(h+1)*d_k
    let heads = |t: Tensor<B, 3>| t.reshape([batch, seq, N_HEADS, d_k]).swap_dims(1, 2);
    let q = heads(layer.mha.query.forward(x.clone()));
    let k = heads(layer.mha.key.forward(x.clone()));
    let v = heads(layer.mha.value.forward(x.clone()));
    let scores = q.matmul(k.transpose()).div_scalar((d_k as f32).sqrt());
    let context = softmax(scores, 3).matmul(v); //                   [batch, heads, seq, d_k]
    let context = context.swap_dims(1, 2).reshape([batch, seq, D_MODEL]);
    let u = layer_norm(x + layer.mha.output.forward(context), &layer.norm_1);
    let ff = layer
        .pwff
        .linear_outer
        .forward(gelu(layer.pwff.linear_inner.forward(u.clone())));
    layer_norm(u + ff, &layer.norm_2)
}

/// Returns (encoder output, masked encoder output, max|by hand - layer 0|, num_params).
fn build() -> (Vec<f32>, Vec<f32>, f32, usize) {
    let device = Default::default();
    let enc = encoder(&device);
    let x = input(&device);

    let out = enc.forward(TransformerEncoderInput::new(x.clone()));
    println!("input {:?} -> output {:?}", x.dims(), out.dims());

    // Sequence 1 is only 3 tokens long: positions 3 and 4 are padding (true = ignore).
    let pad = Tensor::<B, 2, Bool>::from_data(
        TensorData::from([[false; SEQ], [false, false, false, true, true]]),
        &device,
    );
    let masked = enc.forward(TransformerEncoderInput::new(x.clone()).mask_pad(pad));

    let by_layer = enc.layers[0].forward(x.clone(), None, None);
    let by_hand = layer_by_hand(&enc.layers[0], x);
    let diff = (by_layer - by_hand).abs().max().into_scalar();

    (
        out.into_data().to_vec().unwrap(),
        masked.into_data().to_vec().unwrap(),
        diff,
        enc.num_params(),
    )
}

fn fmt(v: &[f32]) -> String {
    let s: Vec<String> = v.iter().map(|x| format!("{x:.4}")).collect();
    format!("[{}]", s.join(", "))
}

fn main() {
    let (out, masked, diff, params) = build();
    println!("seq 0, token 0 = {}", fmt(&out[0..8]));
    println!("seq 1, token 4 = {}", fmt(&out[72..80]));
    println!(
        "with padding mask, seq 1, token 0 = {}",
        fmt(&masked[40..48])
    );
    println!("layer 0 by hand vs Burn's layer: max |diff| = {diff:.1e}");
    println!("num_params = {params}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (out, masked, diff, params) = build();
        // From main.py: nn.TransformerEncoder with the same weights, model.eval().
        let expected: [f32; 80] = [
            -1.900696, -0.776611, -0.089673, 0.574809, 1.192954, 1.009292, 0.309706, -0.235800,
            1.902933, 1.080650, -0.038460, -0.649469, -0.752175, -0.916261, -0.796297, -0.083141,
            -1.835049, -1.046998, -0.414450, 0.193011, 0.799110, 0.959245, 0.791732, 0.594326,
            1.763327, 1.210660, 0.359623, -0.261842, -0.560472, -0.911393, -1.041491, -0.745952,
            -1.651943, -1.100640, -0.569386, -0.040457, 0.534989, 0.861169, 0.976852, 0.996408,
            1.509552, 1.223787, 0.614345, 0.050147, -0.336066, -0.819432, -1.170676, -1.203137,
            -1.532384, -1.104985, -0.637037, -0.148770, 0.403432, 0.799949, 1.041690, 1.166564,
            1.296498, 1.193802, 0.742690, 0.235523, -0.172030, -0.727866, -1.218533, -1.445673,
            -1.344964, -1.076711, -0.736314, -0.306242, 0.249662, 0.714350, 1.086424, 1.372861,
            1.055725, 1.140356, 0.830514, 0.401221, 0.012618, -0.610841, -1.248665, -1.643839,
        ];
        let expected_masked: [f32; 80] = [
            -1.900696, -0.776611, -0.089673, 0.574809, 1.192954, 1.009292, 0.309707, -0.235800,
            1.902933, 1.080650, -0.038460, -0.649469, -0.752175, -0.916261, -0.796297, -0.083141,
            -1.835049, -1.046998, -0.414450, 0.193011, 0.799110, 0.959245, 0.791732, 0.594326,
            1.763327, 1.210660, 0.359623, -0.261842, -0.560472, -0.911393, -1.041491, -0.745952,
            -1.651943, -1.100640, -0.569386, -0.040457, 0.534989, 0.861169, 0.976852, 0.996408,
            1.483188, 1.221516, 0.630854, 0.076267, -0.312558, -0.807389, -1.180475, -1.238461,
            -1.553754, -1.107697, -0.623729, -0.127084, 0.422228, 0.809563, 1.034302, 1.138179,
            1.264892, 1.186993, 0.756765, 0.260942, -0.147899, -0.713009, -1.223629, -1.476012,
            -1.370672, -1.082477, -0.724793, -0.284766, 0.269875, 0.726278, 1.081759, 1.347557,
            1.019168, 1.129073, 0.842076, 0.425803, 0.037216, -0.593339, -1.249122, -1.668994,
        ];
        for (a, b) in out.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        for (a, b) in masked.iter().zip(expected_masked) {
            assert!((a - b).abs() < 1e-5, "masked {a} vs {b}");
        }
        assert!(diff < 1e-5);
        assert_eq!(params, 1200);
    }
}
