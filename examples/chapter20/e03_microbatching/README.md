# Example 20.3 — Micro-batching

Two servers under the same load (32 clients x 40 requests): one forward pass per request vs a queue + worker thread that runs up to 32 requests per batch (max 2 ms wait).

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 20 -e 3 rust      # from the repo root
cargo run -- -c 20 -e 3 python
cargo test -p c20e3 --release     # the Rust test
pytest examples/chapter20/e03_microbatching/python/main.py
```

## Parity

Timings are machine-dependent; both tests assert all 1,280 answers are correct.
