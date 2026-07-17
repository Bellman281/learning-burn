# Example 2.3 — Reshaping and Slicing

Reshape a flat range into a matrix, then flatten, slice, narrow, and gather rows.

```rust
let m = t.reshape([3, 4]);
let piece = m.clone().slice([0..2, 1..3]);   // rows 0-1, cols 1-2
let col = m.clone().narrow(1, 0, 1);         // first column
let rows = m.select(0, indices);             // gather rows 0 and 2
```

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent + `pytest` test. |

## Run & test

```bash
cargo run  --example c2e3
cargo test --example c2e3
python python.py
pytest python.py
```

## Parity

Both sides assert: `reshape` to `[3,4]`, `flatten` to `[12]`, slice `[[1,2],[5,6]]`,
first column `[0,4,8]`, gathered rows `[[0,1,2,3],[8,9,10,11]]` — all int64.
