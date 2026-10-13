# Example 11.6 — Gradient-Checking a Convolution

Chapter 9's central-difference check applied to a convolution with padding 1 and the smooth loss `L = sum(R * tanh(y))`, all in f64. Every one of the 88 gradient entries (2 biases, 36 weights, 50 inputs) is nudged by ±1e-5.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 11 -e 6 rust      # from the repo root
cargo run -- -c 11 -e 6 python
cargo test -p c11e6 --release     # the Rust parity test
pytest examples/chapter11/e06_conv_gradient_check/python/main.py      # the Python parity test
```

## Parity

The hand-written gradients agree with finite differences to a relative error below 1e-8 and with Burn's autodiff in f64 (tensors created with `(&device, DType::F64)` on `Device::flex().autodiff()`; the device's default float is f32) (and torch autograd) to about 1e-14 or better (exactly, on this run, for the bias and weight gradients). A backward pass that forgets to rotate the kernel fails the check with a relative error of 1.92.
