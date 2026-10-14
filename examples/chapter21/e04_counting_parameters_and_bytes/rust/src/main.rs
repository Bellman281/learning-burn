use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Device;

// Chapter 21 — Model footprint, and the no_std picture.
//
// On a microcontroller, memory is THE constraint. Before deploying you want to
// know: how many parameters, and how many bytes at each precision? Module gives
// you num_params(); the rest is arithmetic.

// A small two-layer MLP so there are real parameters to count.
#[derive(Module, Debug)]
struct Mlp {
    l1: Linear, // 64 -> 32 (weights 64*32 + 32 bias)
    l2: Linear, // 32 -> 10 (weights 32*10 + 10 bias)
}

impl Mlp {
    fn new(device: &Device) -> Self {
        Self {
            l1: LinearConfig::new(64, 32).init(device),
            l2: LinearConfig::new(32, 10).init(device),
        }
    }
}

fn main() {
    let dev = Device::flex();
    let model = Mlp::new(&dev);

    let params = model.num_params();
    println!("parameters   = {params}");
    println!("f32  weights = {} bytes", params * 4);
    println!("f16  weights = {} bytes", params * 2);
    println!("int8 weights = {} bytes (+ tiny per-tensor scale)", params);

    // What actually gets baked into flash: the burnpack image of Example 21.1.
    let image = model.into_record().into_bytes().expect("record to bytes");
    println!("burnpack     = {} bytes (f32 weights + header + padding)", image.len());

    // no_std / embedded notes:
    //  - Burn's crates support no_std: Flex with Burn's default features off.
    //  - Weights are baked into the firmware with include_bytes! (Example 21.1) and
    //    loaded from flash with ModuleRecord::from_bytes — no filesystem.
    //  - The same model code runs on any device (Example 21.2), so you develop on CPU
    //    and deploy to the device unchanged.
    //  - Footprint decides feasibility: int8 quantization (Example 21.3) cuts weight
    //    memory ~4x, often the difference between fitting in SRAM/flash or not.
}
