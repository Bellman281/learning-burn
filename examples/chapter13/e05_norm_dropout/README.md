# Example 13.5 — Normalisation and Dropout Layers

`BatchNorm` in training mode (batch statistics, running-statistics update with momentum 0.1) and in eval mode (running statistics), `LayerNorm` with a learned scale and shift, and `Dropout(0.25)`. A layer behaves as in training when its input lives on an autodiff device *and* its own training flag is on (it is on after `init`). On `Device::flex().autodiff()` BatchNorm uses batch statistics and Dropout is active; `.valid()` returns a copy with autodiff and the training flags off, so BatchNorm uses its running statistics, and Dropout returns its input on a plain `Device::flex()`.

```rust
let y_train = bn.forward(bn_input(&device)); // autodiff device: batch stats
let bn_eval = bn.valid(); //                    flag off, no autodiff: running stats
let y_eval = bn_eval.forward(bn_input(&eval_device()));
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 13 -e 5 rust      # from the repo root
cargo run -- -c 13 -e 5 python
cargo test -p c13e5 --release     # the Rust parity test
pytest examples/chapter13/e05_norm_dropout/python/main.py      # the Python parity test
```

## Parity

The training-mode BatchNorm output and `running_mean` match `nn.BatchNorm2d` to 1e-4. `running_var` does **not**: PyTorch feeds the unbiased batch variance (n-1) into it, Burn the biased one (n), so after one step Burn holds `[1.425, 1.0]` and PyTorch `[1.5, 1.0142857]`. The test asserts both, and `main.py` reproduces Burn's eval output once its `running_var` is set to Burn's value. LayerNorm matches `nn.LayerNorm` to 1e-4. Dropout draws different random masks in the two frameworks, so both tests assert the properties instead: every output is 0 or 1/(1-p), about 75% are kept, and in eval mode the output equals the input.
