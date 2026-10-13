use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, Param, RunningState};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

type Eval = NdArray; //            BatchNorm uses its running statistics
type Train = Autodiff<NdArray>; // BatchNorm uses the batch's statistics

const SIZE: usize = 12;

// --- the data: Example 13.3's four shapes, one per class (identical in main.py) ---
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
    img.iter_mut()
        .for_each(|p| *p += rng.next() as f32 / 32768.0 * 0.25);
    img
}

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
    fn tensor<B: Backend, const D: usize>(
        &mut self,
        shape: [usize; D],
        fan_in: usize,
        device: &B::Device,
    ) -> Tensor<B, D> {
        let n = shape.iter().product();
        let v = self.wave(n, 2.3, 1.0 / (fan_in as f64).sqrt(), 0.0);
        Tensor::from_data(TensorData::new(v, shape), device)
    }
    fn vector<B: Backend>(
        &mut self,
        n: usize,
        scale: f64,
        offset: f64,
        d: &B::Device,
    ) -> Tensor<B, 1> {
        Tensor::from_data(TensorData::new(self.wave(n, 1.7, scale, offset), [n]), d)
    }
}

// --- the building blocks ---
// conv (no bias: the BatchNorm's beta does that job) followed by BatchNorm
#[derive(Module, Debug)]
struct ConvBn<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
}

impl<B: Backend> ConvBn<B> {
    fn new(c_in: usize, c_out: usize, k: usize, stride: usize, device: &B::Device) -> Self {
        let p = k / 2; // 3x3 -> padding 1, 1x1 -> padding 0
        Self {
            conv: Conv2dConfig::new([c_in, c_out], [k, k])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(p, p, p, p))
                .with_bias(false)
                .init(device),
            bn: BatchNormConfig::new(c_out).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.bn.forward(self.conv.forward(x))
    }

    fn fill(&mut self, f: &mut Filler, device: &B::Device) {
        let [c_out, c_in, k, _] = self.conv.weight.dims();
        self.conv.weight = Param::from_tensor(f.tensor([c_out, c_in, k, k], c_in * k * k, device));
        self.bn.gamma = Param::from_tensor(f.vector(c_out, 0.1, 1.0, device));
        self.bn.beta = Param::from_tensor(f.vector(c_out, 0.1, 0.0, device));
        self.bn.running_mean = RunningState::new(f.vector(c_out, 0.1, 0.0, device));
        self.bn.running_var = RunningState::new(f.vector(c_out, 0.25, 1.0, device));
    }
}

// y = relu( F(x) + skip(x) ),  F = conv-BN-relu-conv-BN
#[derive(Module, Debug)]
struct ResidualBlock<B: Backend> {
    a: ConvBn<B>,
    b: ConvBn<B>,
    project: Option<ConvBn<B>>, // 1x1 conv + BN, only when the shape changes
}

impl<B: Backend> ResidualBlock<B> {
    fn new(c_in: usize, c_out: usize, stride: usize, device: &B::Device) -> Self {
        let project = if c_in != c_out || stride != 1 {
            Some(ConvBn::new(c_in, c_out, 1, stride, device))
        } else {
            None
        };
        Self {
            a: ConvBn::new(c_in, c_out, 3, stride, device),
            b: ConvBn::new(c_out, c_out, 3, 1, device),
            project,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let skip = match &self.project {
            Some(p) => p.forward(x.clone()),
            None => x.clone(), // the identity: x itself
        };
        let fx = self.b.forward(relu(self.a.forward(x)));
        relu(fx + skip)
    }

    fn fill(&mut self, f: &mut Filler, device: &B::Device) {
        self.a.fill(f, device);
        self.b.fill(f, device);
        if let Some(p) = self.project.as_mut() {
            p.fill(f, device);
        }
    }
}

#[derive(Module, Debug)]
struct SmallResNet<B: Backend> {
    stem: ConvBn<B>,          // 1 -> 8,  12x12
    block1: ResidualBlock<B>, // 8 -> 8,  12x12, identity skip
    block2: ResidualBlock<B>, // 8 -> 16, 6x6, stride 2, projected skip
    gap: AdaptiveAvgPool2d,   // 6x6 -> 1x1
    head: Linear<B>,          // 16 -> 4 classes
}

impl<B: Backend> SmallResNet<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            stem: ConvBn::new(1, 8, 3, 1, device),
            block1: ResidualBlock::new(8, 8, 1, device),
            block2: ResidualBlock::new(8, 16, 2, device),
            gap: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            head: LinearConfig::new(16, 4).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = relu(self.stem.forward(x)); //      [B, 8, 12, 12]
        let x = self.block1.forward(x); //          [B, 8, 12, 12]
        let x = self.block2.forward(x); //          [B, 16, 6, 6]
        let [batch, channels, _, _] = x.dims();
        let x = self.gap.forward(x).reshape([batch, channels]); // [B, 16]
        self.head.forward(x) //                     [B, 4]
    }

    fn with_fixed_weights(mut self, device: &B::Device) -> Self {
        let mut f = Filler { phase: 0.0 };
        self.stem.fill(&mut f, device);
        self.block1.fill(&mut f, device);
        self.block2.fill(&mut f, device);
        // PyTorch stores Linear as [out, in] = [4, 16]; Burn as [in, out]. Transpose.
        let w: Tensor<B, 2> = f.tensor([4, 16], 16, device);
        self.head.weight = Param::from_tensor(w.transpose());
        self.head.bias = Some(Param::from_tensor(f.vector(4, 0.1, 0.0, device)));
        self
    }
}

fn images<B: Backend>(device: &B::Device) -> Tensor<B, 4> {
    let mut rng = Lcg(42);
    let pixels: Vec<f32> = (0..4).flat_map(|c| make_image(c, &mut rng)).collect();
    Tensor::from_data(TensorData::new(pixels, [4, 1, SIZE, SIZE]), device)
}

/// Returns (eval-mode logits, train-mode logits, num_params, running-stat numbers).
fn build() -> (Vec<f32>, Vec<f32>, usize, usize) {
    let device = Default::default();
    let model = SmallResNet::<Train>::new(&device).with_fixed_weights(&device);

    // Eval mode first: .valid() is a copy on the inner backend, running stats untouched.
    let eval_model: SmallResNet<Eval> = model.valid();
    let eval = eval_model.forward(images::<Eval>(&device));
    // Train mode: every BatchNorm normalises with the statistics of these 4 images.
    let train = model.forward(images::<Train>(&device));

    // Burn's num_params() also counts running_mean and running_var (2 per BN channel).
    let bn_channels = 8 + (8 + 8) + (16 + 16 + 16);
    (
        eval.into_data().to_vec().unwrap(),
        train.into_data().to_vec().unwrap(),
        eval_model.num_params(),
        2 * bn_channels,
    )
}

fn main() {
    let (eval, train, params, running) = build();
    for (class, row) in eval.chunks(4).enumerate() {
        println!("eval  logits, class {class} image: {row:.4?}");
    }
    for (class, row) in train.chunks(4).enumerate() {
        println!("train logits, class {class} image: {row:.4?}");
    }
    println!("num_params() = {params}  ({running} of them running statistics)");
    println!("trainable    = {}", params - running);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (eval, train, params, running) = build();
        // From main.py: the same model in model.eval() and model.train() mode.
        let expected_eval = [
            -0.030468, 0.076172, -0.018825, -0.123167, -0.024731, 0.077487, -0.022925, -0.129586,
            -0.037276, 0.075673, -0.012638, -0.114969, -0.071079, 0.035744, -0.028530, -0.094820,
        ];
        let expected_train = [
            -0.072590, 0.023997, -0.041640, -0.099389, -0.041734, -0.097797, -0.224083, -0.204665,
            -0.067853, 0.000614, -0.075479, -0.118123, -0.091215, -0.111750, -0.191967, -0.150741,
        ];
        for (a, b) in eval.iter().zip(expected_eval) {
            assert!((a - b).abs() < 1e-5, "eval {a} vs {b}");
        }
        for (a, b) in train.iter().zip(expected_train) {
            assert!((a - b).abs() < 1e-5, "train {a} vs {b}");
        }
        assert_eq!(params - running, 5020); // sum(p.numel() for p in model.parameters())
        assert_eq!(running, 144); //           the BatchNorm buffers PyTorch keeps apart
    }
}
