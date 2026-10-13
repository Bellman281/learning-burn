# Example 11.5 — The Convolution Backward Pass by Hand

conv (2→3 channels, 3×3) + bias → 2×2 max-pool → `L = sum(pooled * R)`. The backward pass written out:

* max-pool: each pooled gradient goes to its argmax, the rest of the window gets 0;
* `dL/db` = the upstream gradient summed over batch and positions;
* `dL/dW` = the input correlated with the upstream gradient;
* `dL/dx` = the upstream gradient, padded by k−1, correlated with the 180°-rotated kernel (a full convolution).

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 11 -e 5 rust      # from the repo root
cargo run -- -c 11 -e 5 python
cargo test -p c11e5 --release     # the Rust parity test
pytest examples/chapter11/e05_conv_backward_by_hand/python/main.py      # the Python parity test
```

## Parity

The data are multiples of 1/4, so every sum is exact: the hand-written gradients equal `loss.backward()` in Burn (`Autodiff<NdArray>`) and torch `.backward()` bit for bit (`max|hand - burn| = 0`).
