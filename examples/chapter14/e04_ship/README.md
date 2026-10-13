# Example 14.4 — Shipping the Artifact

The training program writes `config.json` and `model.mpk`; a separate loader rebuilds the model from those two files alone.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 14 -e 4 rust      # from the repo root
cargo run -- -c 14 -e 4 python
cargo test -p c14e4 --release     # the Rust test
pytest examples/chapter14/e04_ship/python/main.py
```

## Parity

Identical logits before and after reloading; all 8 test images classified correctly, as in PyTorch.
