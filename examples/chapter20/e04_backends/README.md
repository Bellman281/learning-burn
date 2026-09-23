# Example 20.4 — Backends

The same generic forward pass on `NdArray` and `Flex`, and on `Wgpu` with `--features gpu`.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 20 -e 4 rust      # from the repo root
cargo run -- -c 20 -e 4 python
cargo test -p c20e4 --release     # the Rust test
pytest examples/chapter20/e04_backends/python/main.py
cargo run --release -p c20e4 --features gpu   # adds the Wgpu row (needs a GPU)
```

## Parity

All backends' logits agree with NdArray's to within 1e-4.
