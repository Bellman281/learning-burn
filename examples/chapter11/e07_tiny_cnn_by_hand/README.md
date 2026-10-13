# Example 11.7 — A Tiny CNN Trained by Hand

conv 1→4 (3×3, padding 1) → relu → 2×2 max-pool → linear 144→4, trained on 400 12×12 "shapes" images (horizontal bar, vertical bar, diagonal, plus) with plain SGD (lr 0.3, batch 16, 10 epochs). No autodiff and no optimizer: every gradient comes from the formulas of Examples 11.4 and 11.5 and Chapter 8.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 11 -e 7 rust      # from the repo root
cargo run -- -c 11 -e 7 python
cargo test -p c11e7 --release     # the Rust parity test
pytest examples/chapter11/e07_tiny_cnn_by_hand/python/main.py      # the Python parity test
```

## Parity

Step 1's loss and gradients match Burn's autodiff (`max|hand - burn|` ≈ 1.5e-8 over all 620 gradients) and torch `.backward()`. The mean loss per epoch matches the PyTorch run to 1e-4 (it prints identically to 6 decimals), and both reach 79/80 on the test set.
