# Example 3.2 — Transpose, Matrix-Vector, Outer Product

```rust
let mt = m.clone().transpose();               // or .t()
let mv = linalg::matvec(m.clone(), v.clone()); // matrix * vector
let d = v.clone().dot(v.clone());              // scalar dot product
let op = linalg::outer(v.clone(), v);          // outer product
```

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent (`m.t()`, `m @ v`, `torch.dot`, `torch.outer`) + `pytest` test. |

## Run & test

```bash
cargo run  --example c3e2
cargo test --example c3e2
python python.py
pytest python.py
```

## Parity

Both sides assert: transpose `[[1,3],[2,4]]`, matvec `[3,7]`, dot `2`,
outer `[[1,1],[1,1]]`.
