# Example 20.4 — Backends (devices)

One forward-pass function, compiled once and called with a different `Device` each time: `Device::flex()`, the deprecated `Device::ndarray()` (kept as the baseline it replaces), and `Device::wgpu(DeviceKind::DefaultDevice)` with `--features gpu`.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 20 -e 4 rust      # from the repo root
cargo run -- -c 20 -e 4 python
cargo test -p c20e4 --release     # the Rust test
pytest examples/chapter20/e04_backends/python/main.py
cargo run --release -p c20e4 --features gpu   # adds the Wgpu row (needs a GPU)
```

## Parity

All devices' logits agree with Flex's to within 1e-4.
