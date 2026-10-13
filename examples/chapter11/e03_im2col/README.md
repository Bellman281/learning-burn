# Example 11.3 — im2col: a Convolution as One Matmul

Unfold every 3×3 patch of the padded input into one column of an `[18, 25]` matrix, read the `[3, 2, 3, 3]` weight as a `[3, 18]` matrix, and the whole convolution is one matmul: `[3, 18] @ [18, 25] -> [3, 25]`, plus the bias. This is how most GPU and CPU convolution kernels are built.

```rust
let (cols, ho, wo) = im2col(&x, [2, 5, 5], 3, 1, 1);                  // [18, 25]
let y = (w_mat.clone().matmul(cols_t.clone()) + b).reshape([1, 3, ho, wo]);
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 11 -e 3 rust      # from the repo root
cargo run -- -c 11 -e 3 python
cargo test -p c11e3 --release     # the Rust parity test
pytest examples/chapter11/e03_im2col/python/main.py      # the Python parity test
```

## Parity

The hand-built column matrix equals `burn::tensor::module::unfold4d` and `F.unfold`; the matmul equals `conv2d` and `F.conv2d`, exactly. It also prints the shapes and multiply-adds of a 64→64, 3×3 layer on 56×56 (115,605,504 multiply-adds, a 576×3136 column matrix).
