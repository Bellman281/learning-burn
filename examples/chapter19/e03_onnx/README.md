# Example 19.3 — Code Generated from ONNX

`build.rs` runs `burn_onnx::ModelGen` on `../weights/small_cnn.onnx`, generating Burn source and a `.bpk` weights file at build time; `main.rs` `include!`s the generated model.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 19 -e 3 rust      # from the repo root
cargo run -- -c 19 -e 3 python
cargo test -p c19e3 --release     # the Rust test
pytest examples/chapter19/e03_onnx/python/main.py
```

## Parity

Logits within 1e-5 of PyTorch's; onnxruntime (python/main.py) agrees with PyTorch too.
