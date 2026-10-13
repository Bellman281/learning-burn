# Vector Multiplication: matvec, dot, outer

PyTorch vs Burn: three ways to multiply with a vector. `linalg::matvec` pushes a
vector through a matrix, `dot` collapses two vectors into a scalar, and `outer` grows
an axis to produce a matrix.

`matvec` and `outer` live in `burn::linalg` — they are imported functions, not methods.
In Burn 0.22 that module comes from the `burn-linalg` crate, so the example enables the
`"linalg"` feature next to `"flex"` in its `Cargo.toml`. (`burn::tensor::linalg` still
works, but Burn marks it as a compatibility path.)

> Operations that only make sense at specific ranks live in `linalg`. It keeps the
> `Tensor` method list honest and makes the shape semantics explicit.
