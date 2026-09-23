# Example 18.3 — Checkpoints and Resuming

Train 8 epochs with a file checkpointer, stop, resume from `checkpoint(8)` in a new run, finish at 15.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 18 -e 3 rust      # from the repo root
cargo run -- -c 18 -e 3 python
cargo test -p c18e3 --release     # the Rust test
pytest examples/chapter18/e03_checkpoints/python/main.py
```

## Parity

The resumed run's weights equal an uninterrupted 15-epoch run exactly (difference 0), as with `torch.save`/`torch.load` of model + optimiser state.
