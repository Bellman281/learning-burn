# Example 2.5 — Standardize Features

Z-score standardisation: subtract the per-feature mean, divide by the per-feature
standard deviation.

```rust
let mean = x.clone().mean_dim(0);
let std = x.clone().var(0).sqrt();
let z = (x - mean) / std;
```

## The variance gotcha (that isn't one here)

`var` has two conventions — biased (`/n`) and unbiased (`/(n-1)`, "Bessel's
correction"). **Burn's `var` and PyTorch's `var` both default to unbiased**, so
they agree (`std = [2, 2]`). If you ever see a mismatch on standardisation, this
is the first thing to check — Burn's biased version is `var_bias`, torch's is
`var(..., unbiased=False)`.

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent + `pytest` test. |

## Run & test

```bash
cargo run  --example c2e5
cargo test --example c2e5
python python.py
pytest python.py
```

## Parity

Both sides assert: mean `[3, 4]`, std `[2, 2]`, standardised
`[[-1,-1],[0,0],[1,1]]`.
