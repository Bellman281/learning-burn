# Example 11.2 — 2D Convolution with Loops

A multi-channel 2-D cross-correlation written as seven nested loops over flat `Vec<f32>` buffers, with stride, zero padding and dilation. The input is `[1, 2, 5, 5]`, the weight `[3, 2, 3, 3]`, and every value is a small integer from an LCG, so every output can be checked by hand.

```rust
let r = (i * stride + a * dilation) as isize - padding as isize; // where tap (a, e) lands
let q = (j * stride + e * dilation) as isize - padding as isize;
if r < 0 || q < 0 || r >= h as isize || q >= wd as isize { continue; } // a padding zero
acc += w[((o * ci + c) * kh + a) * kw + e] * x[((b * ci + c) * h + r) * wd + q];
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 11 -e 2 rust      # from the repo root
cargo run -- -c 11 -e 2 python
cargo test -p c11e2 --release     # the Rust parity test
pytest examples/chapter11/e02_conv2d_with_loops/python/main.py      # the Python parity test
```

## Parity

Four settings (stride 1; padding 1; stride 2 with padding 1; dilation 2 with padding 2). The loops, `burn::tensor::module::conv2d` and `F.conv2d` give the same shapes as the output-size rule and the same values, exactly (integer data). The model has 3·2·3·3 + 3 = 57 parameters.
