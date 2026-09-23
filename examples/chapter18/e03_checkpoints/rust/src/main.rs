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
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
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

type MyBackend = Autodiff<NdArray>;

fn loaders(
    config: &TrainingConfig,
) -> (
    std::sync::Arc<dyn burn::data::dataloader::DataLoader<MyBackend, ShapeBatch<MyBackend>>>,
    std::sync::Arc<dyn burn::data::dataloader::DataLoader<NdArray, ShapeBatch<NdArray>>>,
) {
    let train = DataLoaderBuilder::new(ShapeBatcher)
        .batch_size(config.batch_size)
        .build(shapes_dataset(40, 42));
    let valid = DataLoaderBuilder::new(ShapeBatcher)
        .batch_size(config.batch_size)
        .build(shapes_dataset(20, 7));
    (train, valid)
}

/// One run of the Learner in `dir`, up to `epochs`, optionally resuming from a checkpoint.
fn run(dir: &Path, epochs: usize, resume_from: Option<usize>) -> SmallCnn<NdArray> {
    let device = Default::default();
    let config = TrainingConfig::new(SmallCnnConfig::new(), AdamConfig::new().with_epsilon(1e-8));
    let (train, valid) = loaders(&config);
    let mut training = SupervisedTraining::new(dir, train, valid)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_file_checkpointer(NamedMpkFileRecorder::<FullPrecisionSettings>::new())
        .num_epochs(epochs)
        .renderer(Quiet);
    if let Some(epoch) = resume_from {
        training = training.checkpoint(epoch); // load model, optimiser AND scheduler state
    }
    let model = config
        .model
        .init::<MyBackend>(&device)
        .with_fixed_weights(&device);
    let learner = Learner::new(model, config.optimizer.init(), config.learning_rate);
    training.launch(learner).model
}

fn weights(model: &SmallCnn<NdArray>) -> Vec<f32> {
    model.conv1.weight.val().into_data().to_vec().unwrap()
}

/// Returns (checkpoint files after the first run, max |difference| straight vs resumed).
fn build() -> (Vec<String>, f32) {
    let tmp = std::env::temp_dir();
    let straight_dir = tmp.join("learning-burn-c18e3-straight");
    let resumed_dir = tmp.join("learning-burn-c18e3-resumed");
    std::fs::remove_dir_all(&straight_dir).ok();
    std::fs::remove_dir_all(&resumed_dir).ok();

    // A: fifteen epochs in one go.
    let straight = run(&straight_dir, 15, None);

    // B: eight epochs, "crash", then a NEW run that resumes from epoch 8 and finishes.
    run(&resumed_dir, 8, None);
    let mut files: Vec<String> = std::fs::read_dir(resumed_dir.join("checkpoint"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    let resumed = run(&resumed_dir, 15, Some(8));

    let diff = weights(&straight)
        .iter()
        .zip(weights(&resumed))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    (files, diff)
}

fn main() {
    let (files, diff) = build();
    println!("checkpoint/ after 8 epochs: {files:?}");
    println!("15 epochs straight vs 8 + resume + 7: max |weight difference| = {diff:e}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resume_is_exact_like_pytorch() {
        let (files, diff) = build();
        assert!(files.iter().any(|f| f == "model-8.mpk"));
        assert!(files.iter().any(|f| f == "optim-8.mpk"));
        assert_eq!(diff, 0.0); // python/main.py shows the same for torch.save / torch.load
    }
}
