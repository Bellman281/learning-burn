# Example 13.6 — Residual Blocks and a Small ResNet

A `ResidualBlock` -- conv-BN-ReLU-conv-BN plus a skip, with a 1x1 conv + BN projection when the channels or the stride change -- and a small ResNet (stem 1->8, block 8->8, block 8->16 stride 2, global average pool, linear 16->4), all as `#[derive(Module)]`. Every weight, BatchNorm scale/shift and running statistic is set from a fixed formula that `main.py` reproduces, and the logits are compared on Example 13.3's four shapes images.

```rust
#[derive(Module, Debug)]
struct ResidualBlock {
    a: ConvBn,
    b: ConvBn,
    project: Option<ConvBn>, // 1x1 conv + BN, only when the shape changes
}
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 13 -e 6 rust      # from the repo root
cargo run -- -c 13 -e 6 python
cargo test -p c13e6 --release     # the Rust parity test
pytest examples/chapter13/e06_resnet/python/main.py      # the Python parity test
```

## Parity

The 16 logits match PyTorch to 1e-5 in eval mode (`model.eval()` / Burn's `model.valid()`) and in training mode (`model.train()` / the model on `Device::flex().autodiff()`). PyTorch counts 5,020 parameters; Burn's `num_params()` reports 5,164, because it also counts the 144 BatchNorm running statistics that PyTorch keeps as buffers.
