# Example 19.4 — Convert Once

A one-off converter from the PyTorch file to a Burn record, and a loader that needs only `burn`.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 19 -e 4 rust      # from the repo root
cargo run -- -c 19 -e 4 python
cargo test -p c19e4 --release     # the Rust test
pytest examples/chapter19/e04_convert_once/python/main.py
```

## Parity

Logits identical after the round trip, and within 1e-5 of PyTorch's.
