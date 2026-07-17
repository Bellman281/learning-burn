# Example 5.3 — Inner / Inference

There is no `no_grad` block in Burn. For inference you either don't use the
`Autodiff` wrapper at all, or call `inner()` to drop to the base backend and skip
graph tracking. Inference in Burn is a **type**, not a mode.

```rust
let plain = x.inner();          // Autodiff<NdArray> -> NdArray, no tracking
let y = plain.add_scalar(5.0);
```

PyTorch's closest analogue is `with torch.no_grad():` (or `.detach()`).

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn `.inner()` + `#[test]`. |
| `python.py` | PyTorch `torch.no_grad()` + `pytest` test. |

## Run & test

```bash
cargo run  --example c5e3
cargo test --example c5e3
python python.py
pytest python.py
```

## Parity

Both sides assert `[6, 7, 8]`.
