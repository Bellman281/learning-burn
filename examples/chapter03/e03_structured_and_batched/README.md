# Example 3.3 — Structured Matrices and Batched Matmul

Identity, triangular parts, trace, and a batched matrix product.

```rust
let lo = m.clone().tril(0);   // lower-triangular
let up = m.clone().triu(0);   // upper-triangular
let tr = linalg::trace(m);    // sum of the diagonal
let bc = ba.matmul(bb);       // [8,2,3] @ [8,3,4] -> [8,2,4]
```

Batched matmul multiplies matching stacks of matrices: the leading dimension is
the batch, the last two are the matrices.

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent (`tril`, `triu`, `trace`, batched `@`) + `pytest` test. |

## Run & test

```bash
cargo run  --example c3e3
cargo test --example c3e3
python python.py
pytest python.py
```

## Parity

Both sides assert: 3×3 identity, `tril = [[1,0],[3,4]]`, `triu = [[1,2],[0,4]]`,
`trace = 5`, batched product shape `[8, 2, 4]` (all zeros).
