# Example 13.7 — A Transformer Encoder with burn::nn

`TransformerEncoderConfig::new(8, 16, 2, 2)` -- two post-norm layers, two heads, GELU -- with fixed weights copied into PyTorch's `nn.TransformerEncoder` (Burn's separate `query`/`key`/`value` projections become PyTorch's single `in_proj_weight`). Layer 0 is also written out by hand, Chapter 10 style, from the same weights.

```rust
let enc = TransformerEncoderConfig::new(D_MODEL, D_FF, N_HEADS, N_LAYERS).init(device);
let out = enc.forward(TransformerEncoderInput::new(x.clone()));
let masked = enc.forward(TransformerEncoderInput::new(x.clone()).mask_pad(pad));
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 13 -e 7 rust      # from the repo root
cargo run -- -c 13 -e 7 python
cargo test -p c13e7 --release     # the Rust parity test
pytest examples/chapter13/e07_transformer_encoder/python/main.py      # the Python parity test
```

## Parity

All 80 outputs match `nn.TransformerEncoder` (`batch_first=True`, `activation='gelu'`, `eval()`) to 1e-5, with and without a key-padding mask. The hand-written layer agrees with Burn's `TransformerEncoderLayer` to 1e-5. Both frameworks count 1,200 parameters.
