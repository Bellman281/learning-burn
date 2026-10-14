# Example 13.4 — Training the CNN

160 training images go through Chapter 12's `Dataset` → `Batcher` → `DataLoader` in batches of 16. Training is 15 epochs of Adam (lr 0.02, eps 1e-8 to match PyTorch), and the test is 80 held-out images drawn with a different seed.

```rust
let device = Device::flex().autodiff(); // training needs gradients
let loader = DataLoaderBuilder::<ShapeItem, ShapeBatch>::new(ShapeBatcher)
    .batch_size(16)
    .set_device(device.clone())
    .build(train);
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 13 -e 4 rust      # from the repo root
cargo run -- -c 13 -e 4 python
cargo test -p c13e4 --release     # the Rust parity test
pytest examples/chapter13/e04_train_shapes/python/main.py      # the Python parity test
```

## Parity

The mean loss per epoch matches PyTorch to within 2e-3 (in practice to the 4th decimal), and both reach 80/80 on the test set. No shuffling here, so the batch order is identical: Chapter 12 covers shuffling.
