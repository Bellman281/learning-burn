# Example 5.2 — The Gradient Container

PyTorch stores each gradient on the tensor as `.grad`. Burn's `backward()`
instead **returns** all gradients in a container, and you look each one up by
tensor — there is no `.grad` field and therefore no `zero_grad()` to forget.

```rust
let grads = loss.backward();
let da = a.grad(&grads).unwrap();
let db = b.grad(&grads).unwrap();
```

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn autodiff + `#[test]`. |
| `python.py` | PyTorch autograd + `pytest` test. |

## Run & test

```bash
cargo run  --example c5e2
cargo test --example c5e2
python python.py
pytest python.py
```

## Parity

`L = sum(a·b) = 23`; both engines give `dL/da = b = [4, 5]` and
`dL/db = a = [2, 3]`.
