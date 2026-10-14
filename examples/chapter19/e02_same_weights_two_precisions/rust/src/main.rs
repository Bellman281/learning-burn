use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor};
use burn_store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot};
// Chapter 19 — Precision vs file size.
//
// A burnpack file stores each weight in whatever dtype the model holds: f32
// here. An adapter on the store can change that on the way out. Two stores that
// differ only in that adapter:
//      * BurnpackStore -> f32: exact weights
//      * BurnpackStore + HalfPrecisionAdapter -> f16: weights are rounded to
//        half precision on save, and widened back to f32 on load, so
//        predictions shift slightly
//
// Saving the SAME model both ways shows the precision trade-off directly: full
// precision reloads bit-for-bit; half loses a little accuracy. f16 also halves
// the WEIGHT bytes -- but that only shrinks the file on a real model with
// thousands of parameters. On this 2-parameter toy the file is almost all
// header and alignment padding (see the printout).
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
    let dev = Device::flex();

    let x = Tensor::<2>::from_floats([[2.0]], &dev);
    let model = Model::new(&dev);
    let original = model.forward(x.clone());

    // --- Full precision (f32): exact round-trip ---
    let full_path = std::env::temp_dir().join("ch19_full.bpk");
    let mut full = BurnpackStore::from_file(&full_path).overwrite(true);
    model.save_into(&mut full).expect("save full");
    let mut from_full = Model::new(&dev);
    from_full
        .load_from(&mut BurnpackStore::from_file(&full_path))
        .expect("load full");
    let pred_full = from_full.forward(x.clone());

    // --- Half (f16): rounded on the way out, back to f32 on the way in ---
    let half_path = std::env::temp_dir().join("ch19_half.bpk");
    let mut half = BurnpackStore::from_file(&half_path)
        .overwrite(true)
        .with_to_adapter(HalfPrecisionAdapter::new());
    model.save_into(&mut half).expect("save half");
    let mut from_half = Model::new(&dev);
    from_half
        .load_from(
            &mut BurnpackStore::from_file(&half_path)
                .with_from_adapter(HalfPrecisionAdapter::new()),
        )
        .expect("load half");
    let pred_half = from_half.forward(x);

    // Without the adapter, a load keeps the dtype that is in the file.
    let as_stored = Model::new(&dev).load_file(&half_path);

    let full_size = std::fs::metadata(&full_path).map(|m| m.len()).unwrap_or(0);
    let half_size = std::fs::metadata(&half_path).map(|m| m.len()).unwrap_or(0);
    println!("original prediction   = {}", original.to_data());
    println!("full-precision reload = {}", pred_full.to_data()); // exactly equal
    println!("half (f16)     reload = {}", pred_half.to_data()); // very close
    println!("full (f32)     file   = {full_size} bytes");
    println!("half (f16)     file   = {half_size} bytes");
    println!(
        "half file, plain load_file: weight dtype = {:?}",
        as_stored.linear.weight.val().dtype()
    );
}
