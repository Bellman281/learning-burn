# Example 2.6 — Element-wise Masking

Transcendental ops chained, clamping, a boolean mask, and a masked select.

```rust
let y = x.clone().exp().log1p();          // softplus
let c = x.clone().clamp(0.0, 1.0);
let mask = x.clone().greater_elem(0.0);   // Bool tensor
let picked = x.mask_where(mask, replacement);
```

`mask_where` puts the replacement where the mask is **true** and keeps the
original where false — exactly `torch.where(mask, value, x)`.

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent + `pytest` test. |

## Run & test

```bash
cargo run  --example c2e6
cargo test --example c2e6
python python.py
pytest python.py
```

## Parity

Both sides assert: softplus `[0.3133, 0.6931, 2.1269]` (tol 1e-4), clamp `[0, 0, 1]`,
masked select `[-1, 0, 9]`.
