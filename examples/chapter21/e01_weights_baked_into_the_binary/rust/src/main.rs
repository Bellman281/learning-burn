use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::store::ModuleRecord;
use burn::tensor::{Device, Tensor};

// Chapter 21 — Bare-metal inference: weights embedded in the binary.
//
// A microcontroller has no filesystem. So instead of load_file(), you serialize
// the weights to a byte buffer, bake that buffer into the firmware with
// `include_bytes!("model.bpk")`, and load it back from flash — no disk involved.
// ModuleRecord does exactly this: into_bytes() -> burnpack bytes,
// from_bytes(bytes) -> weights.

#[derive(Module, Debug)]
struct Model {
    linear: Linear,
}
impl Model {
    fn new(device: &Device) -> Self {
        Self {
            linear: LinearConfig::new(4, 1).init(device),
        }
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.linear.forward(x)
    }
}
fn main() {
    let device = Device::flex();

    let x = Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
    let model = Model::new(&device);
    let before = model.forward(x.clone());

    // Serialize weights to an in-memory burnpack buffer — NO filesystem.
    let bytes = model.into_record().into_bytes().expect("record to bytes");
    println!("model serialized to {} bytes (held in RAM)", bytes.len());

    // On-device these bytes come from include_bytes!("model.bpk"), baked into the
    // firmware. Load them back with zero file I/O:
    let record = ModuleRecord::from_bytes(bytes.clone()).expect("load from bytes");
    let loaded = Model::new(&device).load_record(record);

    let after = loaded.forward(x);

    println!("prediction before = {}", before.to_data());
    println!("prediction after  = {}", after.to_data()); // identical
}
