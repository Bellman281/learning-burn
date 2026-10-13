use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, SgdConfig};
use burn::tensor::{Device, Tensor};

// A minimal model: a single Linear(1 -> 1) layer, i.e. y = w*x + b.
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
    // Training needs gradients, so the device has autodiff switched on.
    let device = Device::flex().autodiff();

    // Data generated from y = 2x + 1.
    let x = Tensor::<2>::from_floats([[1.0], [2.0], [3.0], [4.0]], &device);
    let y = Tensor::<2>::from_floats([[3.0], [5.0], [7.0], [9.0]], &device);

    let mut model = Model::new(&device);
    let mut optim = SgdConfig::new().init();
    let lr = 0.02;

    // The four-beat training loop: forward -> loss -> backward -> step.
    for _ in 0..3000 {
        let pred = model.forward(x.clone());
        let loss = MseLoss::new().forward(pred, y.clone(), Reduction::Mean);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }

    // Should now approximate y = 2x + 1, so f(5) ~ 11.
    let test = Tensor::<2>::from_floats([[5.0]], &device);
    println!("prediction for x=5: {}", model.forward(test).to_data()); // ~11
}
