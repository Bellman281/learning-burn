# Example 13.1 — Burn's Conv2d, Next to Chapter 11's Function

The vertical-edge kernel run through `burn::tensor::module::conv2d` with the weight passed in (Chapter 11's way) and through an `nn::conv::Conv2d` layer that owns the same weight as a `Param`. Then the default initialisation bound, and the output-size rule checked against the layer for padding, stride and dilation.

```rust
let mut layer = Conv2dConfig::new([1, 1], [3, 3])
    .with_bias(false)
    .init(&device);
layer.weight = Param::from_tensor(kernel); // replace the random init
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 13 -e 1 rust      # from the repo root
cargo run -- -c 13 -e 1 python
cargo test -p c13e1 --release     # the Rust parity test
pytest examples/chapter13/e01_conv2d_layer/python/main.py      # the Python parity test
```

## Parity

Function and layer both give the edge map `[3, 3, 0]` on every row, as `F.conv2d` and `nn.Conv2d` do. Both frameworks' default init stays inside +-1/sqrt(fan_in). The four 12x12 cases give 10, 12, 6 and 8, and the weight shape is `[8, 1, 3, 3]` in both.
