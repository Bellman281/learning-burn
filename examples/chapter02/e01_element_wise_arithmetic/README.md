# Element-wise arithmetic

PyTorch vs Burn: addition, multiplication, scalar multiply, and negation—all applied
slot by slot. Burn consumes operands, so reuse requires `.clone()`.

Scalar multiplication uses the named method `mul_scalar`. The operator form `a * 2.0`
compiles too and calls the same operation; the named method just says on the page that
one side is a plain number.

> `a * b` is element-wise, not matrix multiplication. For the matrix product, use
> `matmul` — exactly the same pitfall as NumPy's `*` vs `@`.
