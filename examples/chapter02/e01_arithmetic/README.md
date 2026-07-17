# Example 2.1 — Element-wise Arithmetic

Operators (`+`, `*`, `-`) and their scalar variants. `*` is **element-wise**, not
matrix multiplication. Reused tensors are `.clone()`d.

```rust
let s = a.clone() + b.clone();
let p = a.clone() * b.clone();     // element-wise
let sc = a.clone().mul_scalar(2.0);
let neg = -a;
```

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent + `pytest` test. |

## Run & test

```bash
cargo run  --example c2e1
cargo test --example c2e1
python python.py
pytest python.py
```

## Parity

Both sides assert: sum `[11,22,33]`, product `[10,40,90]`, scaled `[2,4,6]`,
negation `[-1,-2,-3]`.
