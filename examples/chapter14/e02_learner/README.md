# Example 14.2 — The Learner

Chapter 13's CNN trained by `SupervisedTraining` + `Learner` through `TrainStep`/`InferenceStep`, with loss and accuracy metrics read back from the Learner's log files.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 14 -e 2 rust      # from the repo root
cargo run -- -c 14 -e 2 python
cargo test -p c14e2 --release     # the Rust test
pytest examples/chapter14/e02_learner/python/main.py
```

## Parity

The per-epoch train loss, validation loss and validation accuracy match PyTorch's hand-written loop (python/main.py) to within 2e-3; printed to 4 decimals they are identical.
