# Example 19.4 — safetensors and the Transpose

`SafetensorsStore` with and without `PyTorchToBurnAdapter`: without it, the Linear weight's `[4, 8]` vs `[8, 4]` shape mismatch stops the load.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 19 -e 2 rust      # from the repo root
cargo run -- -c 19 -e 2 python
cargo test -p c19e4 --release     # the Rust test
pytest examples/chapter19/e04_safetensors/python/main.py
```

## Parity

Logits within 1e-5 of PyTorch's; same 8 predictions.
