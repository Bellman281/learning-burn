use burn::backend::{Autodiff, NdArray};
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::InMemDataset;
use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::optim::AdamConfig;
use burn::tensor::activation::relu;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use burn::train::metric::{AccuracyMetric, LossMetric, MetricDefinition};
use burn::train::renderer::{
    EvaluationName, EvaluationProgress, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    MetricsRendererTraining, ProgressType, TrainingProgress,
};
use burn::train::{
    ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
};
use std::path::Path;

const SIZE: usize = 12;

// --- the data: 12x12 "shapes", generated in code (identical in python.py) ---
// A tiny LCG so Rust and Python draw exactly the same "random" numbers.
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

// --- the data pipeline, exactly as in Chapter 15 ---
#[derive(Clone, Debug)]
struct ShapeItem {
    pixels: Vec<f32>,
    label: usize,
}

// Class order cycles 0,1,2,3,0,1,... so every batch of 16 is balanced.
fn shapes_dataset(per_class: usize, seed: u32) -> InMemDataset<ShapeItem> {
    let mut rng = Lcg(seed);
    let items = (0..per_class * 4)
        .map(|i| ShapeItem {
            pixels: make_image(i % 4, &mut rng),
            label: i % 4,
        })
        .collect();
    InMemDataset::new(items)
}

#[derive(Clone, Debug)]
struct ShapeBatch<B: Backend> {
    images: Tensor<B, 4>,       // [batch, 1, 12, 12]
    targets: Tensor<B, 1, Int>, // [batch]
}

#[derive(Clone, Default)]
struct ShapeBatcher;

impl<B: Backend> Batcher<B, ShapeItem, ShapeBatch<B>> for ShapeBatcher {
    fn batch(&self, items: Vec<ShapeItem>, device: &B::Device) -> ShapeBatch<B> {
        let n = items.len();
        let pixels: Vec<f32> = items.iter().flat_map(|it| it.pixels.clone()).collect();
        let labels: Vec<i64> = items.iter().map(|it| it.label as i64).collect();
        ShapeBatch {
            images: Tensor::from_data(TensorData::new(pixels, [n, 1, SIZE, SIZE]), device),
            targets: Tensor::from_data(TensorData::new(labels, [n]), device),
        }
    }
}

// --- the model, now built from a Config ---
#[derive(Config, Debug)]
struct SmallCnnConfig {
    #[config(default = 4)]
    channels1: usize,
    #[config(default = 8)]
    channels2: usize,
    #[config(default = 4)]
    classes: usize,
}

impl SmallCnnConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> SmallCnn<B> {
        let same = PaddingConfig2d::Explicit(1, 1, 1, 1);
        SmallCnn {
            conv1: Conv2dConfig::new([1, self.channels1], [3, 3])
                .with_padding(same.clone())
                .init(device),
            pool: MaxPool2dConfig::new([2, 2]).init(),
            conv2: Conv2dConfig::new([self.channels1, self.channels2], [3, 3])
                .with_padding(same)
                .init(device),
            gap: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            head: LinearConfig::new(self.channels2, self.classes).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct SmallCnn<B: Backend> {
    conv1: Conv2d<B>,
    pool: MaxPool2d,
    conv2: Conv2d<B>,
    gap: AdaptiveAvgPool2d,
    head: Linear<B>,
}

impl<B: Backend> SmallCnn<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.pool.forward(relu(self.conv1.forward(x)));
        let x = relu(self.conv2.forward(x));
        let [batch, channels, _, _] = x.dims();
        let x = self.gap.forward(x).reshape([batch, channels]);
        self.head.forward(x)
    }

    // Chapter 15's fixed starting weights, so every run can be checked against PyTorch.
    // Written for the default config (4 and 8 channels, 4 classes).
    fn with_fixed_weights(mut self, device: &B::Device) -> Self {
        let det = |n: usize, fan_in: usize, phase: f64| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f64 * 2.3 + phase).sin() / (fan_in as f64).sqrt()) as f32)
                .collect()
        };
        self.conv1.weight = param(det(36, 9, 0.1), [4, 1, 3, 3], device);
        self.conv1.bias = Some(param(vec![0.0; 4], [4], device));
        self.conv2.weight = param(det(288, 36, 0.3), [8, 4, 3, 3], device);
        self.conv2.bias = Some(param(vec![0.0; 8], [8], device));
        let w = det(32, 8, 0.5);
        let w_t: Vec<f32> = (0..32).map(|k| w[(k % 4) * 8 + k / 4]).collect();
        self.head.weight = param(w_t, [8, 4], device);
        self.head.bias = Some(param(vec![0.0; 4], [4], device));
        self
    }
}

fn param<B: Backend, const D: usize>(
    v: Vec<f32>,
    shape: [usize; D],
    device: &B::Device,
) -> Param<Tensor<B, D>> {
    Param::from_tensor(Tensor::from_data(TensorData::new(v, shape), device))
}

// Everything a training run depends on, in one serialisable value.
#[derive(Config, Debug)]
struct TrainingConfig {
    model: SmallCnnConfig,
    optimizer: AdamConfig,
    #[config(default = 15)]
    num_epochs: usize,
    #[config(default = 16)]
    batch_size: usize,
    #[config(default = 0.02)]
    learning_rate: f64,
}

// --- the two traits the Learner needs: one training step, one inference step ---
impl<B: Backend> SmallCnn<B> {
    fn forward_classification(&self, batch: ShapeBatch<B>) -> ClassificationOutput<B> {
        let logits = self.forward(batch.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), batch.targets.clone());
        ClassificationOutput::new(loss, logits, batch.targets)
    }
}

impl<B: AutodiffBackend> TrainStep for SmallCnn<B> {
    type Input = ShapeBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: ShapeBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for SmallCnn<B> {
    type Input = ShapeBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: ShapeBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch)
    }
}

// A renderer that draws nothing: the Learner's default is a live terminal dashboard,
// which is great interactively and unreadable in a book. We read the logs instead.
struct Quiet;
impl MetricsRendererTraining for Quiet {
    fn update_train(&mut self, _: MetricState) {}
    fn update_valid(&mut self, _: MetricState) {}
    fn render_train(&mut self, _: TrainingProgress, _: Vec<ProgressType>) {}
    fn render_valid(&mut self, _: TrainingProgress, _: Vec<ProgressType>) {}
}
impl MetricsRendererEvaluation for Quiet {
    fn update_test(&mut self, _: EvaluationName, _: MetricState) {}
    fn render_test(&mut self, _: EvaluationProgress, _: Vec<ProgressType>) {}
}
impl MetricsRenderer for Quiet {
    fn manual_close(&mut self) {}
    fn register_metric(&mut self, _: MetricDefinition) {}
}

// The Learner logs every batch to <dir>/<split>/epoch-N/<Metric>.log as "value,count".
// The epoch's number is the count-weighted mean of those lines.
fn epoch_mean(dir: &Path, split: &str, epoch: usize, metric: &str) -> f64 {
    let log = std::fs::read_to_string(dir.join(format!("{split}/epoch-{epoch}/{metric}.log")))
        .expect("metric log");
    let (mut sum, mut n) = (0.0, 0.0);
    for line in log.lines() {
        let (v, c) = line.split_once(',').unwrap();
        let c: f64 = c.parse().unwrap();
        sum += v.parse::<f64>().unwrap() * c;
        n += c;
    }
    sum / n
}

type MyBackend = Autodiff<NdArray>;

/// Trains with the Learner. Returns, per epoch: (train loss, valid loss, valid accuracy %).
fn build() -> Vec<(f64, f64, f64)> {
    let device = Default::default();
    let config = TrainingConfig::new(SmallCnnConfig::new(), AdamConfig::new().with_epsilon(1e-8));
    let dir = std::env::temp_dir().join("learning-burn-c18e2");
    std::fs::remove_dir_all(&dir).ok();

    let train = DataLoaderBuilder::new(ShapeBatcher)
        .batch_size(config.batch_size)
        .build(shapes_dataset(40, 42));
    let valid = DataLoaderBuilder::new(ShapeBatcher)
        .batch_size(config.batch_size)
        .build(shapes_dataset(20, 7));

    let training = SupervisedTraining::new(&dir, train, valid)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .num_epochs(config.num_epochs)
        .renderer(Quiet);

    let model = config
        .model
        .init::<MyBackend>(&device)
        .with_fixed_weights(&device);
    let learner = Learner::new(model, config.optimizer.init(), config.learning_rate);
    let _trained = training.launch(learner).model;

    (1..=config.num_epochs)
        .map(|e| {
            (
                epoch_mean(&dir, "train", e, "Loss"),
                epoch_mean(&dir, "valid", e, "Loss"),
                epoch_mean(&dir, "valid", e, "Accuracy"),
            )
        })
        .collect()
}

fn main() {
    println!("epoch  train loss  valid loss  valid acc");
    for (e, (tl, vl, va)) in build().iter().enumerate() {
        println!("{:>5}  {tl:>10.4}  {vl:>10.4}  {va:>8.1}%", e + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let got = build();
        // From python/main.py: the same data, order, weights and optimiser.
        let train = [
            1.344356, 1.104616, 0.853722, 0.715782, 0.585857, 0.470269, 0.384723, 0.276252,
            0.130639, 0.056581, 0.031400, 0.028396, 0.021925, 0.020555, 0.009121,
        ];
        let valid = [
            1.249453, 0.943234, 0.772485, 0.642903, 0.513178, 0.421801, 0.342713, 0.217261,
            0.105364, 0.055251, 0.035643, 0.018129, 0.021608, 0.010215, 0.012992,
        ];
        let acc = [
            53.7500, 53.7500, 75.0000, 75.0000, 75.0000, 81.2500, 90.0000, 96.2500, 100.0000,
            100.0000, 100.0000, 100.0000, 100.0000, 100.0000, 100.0000,
        ];
        for e in 0..15 {
            assert!((got[e].0 - train[e]).abs() < 2e-3, "train {e}");
            assert!((got[e].1 - valid[e]).abs() < 2e-3, "valid {e}");
            assert!((got[e].2 - acc[e]).abs() < 1e-6, "acc {e}");
        }
    }
}
