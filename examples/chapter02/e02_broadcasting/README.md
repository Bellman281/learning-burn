# Example 2.2 — Broadcasting

Add a per-column bias to every row. PyTorch inserts the missing axis silently;
Burn asks you to be explicit with `unsqueeze()`.

```rust
let out = m + bias.unsqueeze();   // [3] -> [1, 3], then broadcast over 2 rows
```

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent + `pytest` test. |

## Run & test

```bash
cargo run  --example c2e2
cargo test --example c2e2
python python.py
pytest python.py
```

## Parity

Both sides assert `[[101, 202, 303], [104, 205, 306]]`, shape `[2, 3]`.
