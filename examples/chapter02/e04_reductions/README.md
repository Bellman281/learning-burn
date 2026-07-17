# Example 2.4 — Reductions

Sum, per-dimension sum/mean, and argmax-style reduction.

```rust
let total = x.clone().sum();                 // scalar
let col_sum = x.clone().sum_dim(0);          // [[4, 6]]
let row_mean = x.clone().mean_dim(1);        // [[1.5], [3.5]]
let (m, idx) = x.max_dim_with_indices(1);    // values + argmax
```

## A convention difference (not a value difference)

Burn's dim reductions **keep** the reduced axis (`[2, 1]`), while PyTorch's
`max(dim=...)` **drops** it (`[2]`). The *values* are identical — the parity test
compares them flattened.

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent + `pytest` test. |

## Run & test

```bash
cargo run  --example c2e4
cargo test --example c2e4
python python.py
pytest python.py
```

## Parity

Both sides assert: total `10`, column sums `[4, 6]`, row means `[1.5, 3.5]`,
max values `[2, 4]`, argmax indices `[1, 1]`.
