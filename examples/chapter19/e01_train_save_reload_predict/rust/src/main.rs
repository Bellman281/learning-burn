use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, SgdConfig};
use burn::tensor::{Device, Tensor};

// Chapter 19 — Saving and loading: train, save, reload, infer.
//
// Training and inference are usually separate programs. You train once, write
// the weights to disk, then a lightweight inference process loads them and
// serves predictions. Burn writes the weights as a burnpack (.bpk) file; the
// model architecture is just Rust code that both programs share.
//
// Here we do the whole round trip in one file so you can see it end to end.
//
#[derive(Module, Debug)]
struct Model {
    linear: Linear,
}
impl Model {
    fn new(device: &Device) -> Self {
        Self {
            linear: LinearConfig::new(1, 1).init(device),
        }
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.linear.forward(x)
    }
}
fn main() {
    let train_device = Device::flex().autodiff(); // training needs gradients
    let infer_device = Device::flex(); // inference does not
    let path = std::env::temp_dir().join("burn_ch19_model");

    // ---- 1. TRAIN (fit y = 2x + 1) ----
    let x = Tensor::<2>::from_floats([[1.0], [2.0], [3.0], [4.0]], &train_device);
    let y = Tensor::<2>::from_floats([[3.0], [5.0], [7.0], [9.0]], &train_device);
    let mut model = Model::new(&train_device);
    let mut optim = SgdConfig::new().init();
    for _ in 0..3000 {
        let loss = MseLoss::new().forward(model.forward(x.clone()), y.clone(), Reduction::Mean);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(0.02, model, grads);
    }

    // ---- 2. SAVE weights to disk (consumes the trained model) ----
    model.save_file(&path).expect("failed to save model");
    println!("saved weights to {:?}", path.with_extension("bpk"));

    // ---- 3. LOAD into a fresh inference model (plain device, no autodiff) ----
    let loaded = Model::new(&infer_device)
        .try_load_file(&path)
        .expect("failed to load model");

    // ---- 4. INFER ----
    let test = Tensor::<2>::from_floats([[5.0]], &infer_device);
    let pred = loaded.forward(test);

    println!("prediction for x=5 after reload: {}", pred.to_data()); // ~11 (= 2*5 + 1)
    println!(
        "reloaded weight on autodiff: {}",
        loaded.linear.weight.val().is_autodiff()
    );
}
