# Example 18.1 — Configs

`#[derive(Config)]` for the model and the training run: defaults, `with_*` builders, and JSON `save`/`load`. The model is built from its config.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 18 -e 1 rust      # from the repo root
cargo run -- -c 18 -e 1 python
cargo test -p c18e1 --release     # the Rust test
pytest examples/chapter18/e01_config/python/main.py
```

## Parity

Same parameter counts from the reloaded config (372; 700 with `channels2 = 16`).
