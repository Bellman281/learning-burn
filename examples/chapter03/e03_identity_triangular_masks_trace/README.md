# Identity, Triangular Masks & Trace

PyTorch vs Burn: creating an identity with `eye`, extracting lower/upper triangles
with `tril`/`triu`, and summing the diagonal with `linalg::trace` (from the `burn-linalg`
crate, enabled by the `"linalg"` feature).

> The upper-triangular mask (`triu`) is the trick behind causal attention in
> transformers — zeroing out the future so a token cannot see what comes after it.
