# Example 3.1 — Matrix Multiplication

`matmul` is the real matrix product — `[m,k] @ [k,n] -> [m,n]`, inner dims must
match. Element-wise `*` is a different operation.

```rust
let c = a.matmul(b);   // [2,3] @ [3,2] -> [2,2]
```

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent (`a @ b`) + `pytest` test. |

## Run & test

```bash
cargo run  --example c3e1
cargo test --example c3e1
python python.py
pytest python.py
```

## Parity

Both sides assert `[[4, 5], [10, 11]]`, shape `[2, 2]`.
