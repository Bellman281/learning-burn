# Example 5.1 — Backward Gradient

`Autodiff` is a backend **decorator** — wrap any backend to get gradients. Mark
the input with `require_grad()`, call `backward()`, and look the gradient up.

```rust
type Backend = Autodiff<NdArray>;
let x = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device).require_grad();
let f = (x.clone() * x.clone()).sum();
let grads = f.backward();
let dx = x.grad(&grads).unwrap();   // 2x
```

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn autodiff + `#[test]`. |
| `python.py` | PyTorch autograd (`requires_grad`, `.backward()`, `.grad`) + `pytest` test. |

## Run & test

```bash
cargo run  --example c5e1
cargo test --example c5e1
python python.py
pytest python.py
```

## Parity

`f(x) = sum(x²) = 14`, and both engines compute `df/dx = 2x = [2, 4, 6]`
— Burn's autodiff and PyTorch's autograd agree exactly.
