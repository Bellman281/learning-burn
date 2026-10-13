# Example 20.1 — An Inference Server

An axum service: the model (Chapter 19's weights) loaded once into an `Arc`, `POST /predict` with `{"pixels": [...]}`, `GET /health`, HTTP 422 for malformed input. The example starts the server and acts as its own client.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 20 -e 1 rust      # from the repo root
cargo run -- -c 20 -e 1 python
cargo test -p c20e1 --release     # the Rust test
pytest examples/chapter20/e01_server/python/main.py
```

## Parity

Same classes for the 8 test images as the Python `http.server` version.
