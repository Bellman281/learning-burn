# Example 11.4 — Pooling by Hand

Max and average pooling as loops over the windows. Max pooling also records the flat index `r * w + q` of each window's maximum: the backward pass in Example 11.5 needs exactly that. The 6×6 input is a shuffle of 0..35, so no two pixels tie.

```rust
let (bv, bi) = max_pool2d_with_indices(xt.clone(), [k, k], [s, s], [0, 0], [1, 1], false);
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 11 -e 4 rust      # from the repo root
cargo run -- -c 11 -e 4 python
cargo test -p c11e4 --release     # the Rust parity test
pytest examples/chapter11/e04_pooling_by_hand/python/main.py      # the Python parity test
```

## Parity

Values and indices match `burn::tensor::module::max_pool2d_with_indices` and `F.max_pool2d(return_indices=True)`; averages match `avg_pool2d`. With overlapping 3×3 windows (stride 2), the pixel 35 at (2, 2) wins all four windows. Also prints the receptive field of conv 3×3 → pool 2×2 → conv 3×3: 3, 4, 8.
