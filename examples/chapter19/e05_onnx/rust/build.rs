// Runs before the crate compiles: reads the ONNX file and WRITES RUST SOURCE
// for the model (plus a weights file) into OUT_DIR. main.rs include!s it.
use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("../../weights/small_cnn.onnx")
        .out_dir("model/")
        .run_from_script();
    println!("cargo:rerun-if-changed=../../weights/small_cnn.onnx");
}
