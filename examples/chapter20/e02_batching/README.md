# Example 20.2 — Batch Size and Throughput

Median forward-pass time for batch sizes 1 to 256.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 20 -e 2 rust      # from the repo root
cargo run -- -c 20 -e 2 python
cargo test -p c20e2 --release     # the Rust test
pytest examples/chapter20/e02_batching/python/main.py
```

## Parity

Timings are machine-dependent and not asserted; both tests assert that batching does not change the logits.
