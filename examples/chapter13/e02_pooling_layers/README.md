# Example 13.2 — Pooling Layers

Max, average, global-average and adaptive-average pooling layers on a 4x4 image of 1..16, plus the receptive field of the conv -> pool -> conv stack that Example 13.3's CNN uses.

```rust
let max = MaxPool2dConfig::new([2, 2]).init().forward(x.clone());
let avg = AvgPool2dConfig::new([2, 2]).init().forward(x.clone());
let global = AdaptiveAvgPool2dConfig::new([1, 1]).init().forward(x.clone());
let adaptive = AdaptiveAvgPool2dConfig::new([3, 3]).init().forward(x);
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 13 -e 2 rust      # from the repo root
cargo run -- -c 13 -e 2 python
cargo test -p c13e2 --release     # the Rust parity test
pytest examples/chapter13/e02_pooling_layers/python/main.py      # the Python parity test
```

## Parity

Max `[6, 8, 14, 16]`, average `[3.5, 5.5, 11.5, 13.5]`, global `8.5`, adaptive 3x3 `[3.5, 4.5, 5.5, 7.5, 8.5, 9.5, 11.5, 12.5, 13.5]`. The receptive field grows 3 -> 4 -> 8 pixels. All of it is identical in PyTorch.
