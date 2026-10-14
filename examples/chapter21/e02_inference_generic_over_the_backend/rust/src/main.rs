use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor};

// Chapter 21 — Backend-agnostic inference.
//
// Burn's core idea: your model and inference code never name a backend. They
// take a `&Device`, a run-time value. The exact same compiled function runs on
// CPU (Flex), GPU (Wgpu/CUDA), or an embedded CPU — you only pick the device at
// the call site. Which backends exist in the binary at all is decided at compile
// time, by Cargo features.

#[derive(Module, Debug)]
struct Classifier {
    linear: Linear,
}

impl Classifier {
    fn new(device: &Device) -> Self {
        Self {
            linear: LinearConfig::new(3, 2).init(device),
        }
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.linear.forward(x)
    }
}

// Inference written ONCE, for any device. Nothing here mentions a concrete device.
fn run_inference(device: &Device) -> Tensor<2> {
    let model = Classifier::new(device);
    let x = Tensor::<2>::from_floats([[1.0, 2.0, 3.0]], device);
    model.forward(x)
}

fn main() {
    // CPU today:
    let out = run_inference(&Device::flex());

    println!("output shape = {:?}", out.dims()); // [1, 2]
    println!("output = {}", out.to_data());

    // To run the SAME function on a GPU (build with the "wgpu" feature):
    //      run_inference(&Device::wgpu(DeviceKind::DefaultDevice));
    // For embedded, build Flex without std. The function body never changes.
}
