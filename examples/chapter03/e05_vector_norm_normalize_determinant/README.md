# Linear Algebra: Norm, Normalize, and Determinant

This example demonstrates three core linear algebra operations in PyTorch and Burn (Rust):

- **$L_2$ Norm:** Calculates the straight-line length of the vector $[3, 4]$, resulting in $5.0$.
- **Vector Normalization:** Scales the vector to a unit length of $[0.6, 0.8]$. In Burn, this is done via `linalg::vector_normalize` and requires an epsilon (tolerance) parameter.
- **Determinant:** Computes the determinant of a $2 \times 2$ matrix (resulting in $-6.0$).

### Key Differences in Burn (Rust)
- **Batched Determinants:** Burn's `linalg::det` operates on a *batch* of square matrices (minimum rank 3), meaning a $2 \times 2$ matrix must be shaped as $[1, 2, 2]$.
- **Turbofish Generics:** The `linalg::det` function requires explicit rank generics (`det::<D, D1, D2>`, e.g. `det::<3, 2, 1>`) whose values are fully determined by `D` (`D1 = D - 1`, `D2 = D - 2`). Rust cannot compute them in the signature, so you must state them, and Burn checks them at run time: a wrong combination, or an input of rank below 3, panics.
