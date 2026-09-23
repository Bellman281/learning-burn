# Example 19.1 — A PyTorch .pt File

`PytorchStore` loads `../weights/small_cnn.pt` into a Burn struct, with key remapping from PyTorch's names (`features.0`, `features.3`, `classifier`) to Burn's (`conv1`, `conv2`, `head`). The naive load's error report is printed first.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 19 -e 1 rust      # from the repo root
cargo run -- -c 19 -e 1 python
cargo test -p c19e1 --release     # the Rust test
pytest examples/chapter19/e01_pytorch_pt/python/main.py
```

## Parity

Logits within 1e-5 of PyTorch's (`../weights/reference.json`); same 8 predictions.
