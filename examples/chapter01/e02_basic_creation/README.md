# Example 1.2 — Creating Tensors from Data

This example demonstrates the two standard ways to create a tensor in Burn.

## What this example shows

Burn provides two constructors:

```rust
Tensor::from_floats(...)
```

and

```rust
Tensor::from_data(...)
```

Both create the same tensor; they only differ in how the input data is provided.

## Key takeaway

* `from_floats` is the ergonomic choice for floating-point arrays.
* `from_data` is the generic constructor using `TensorData`.
* Every tensor is created on a specific **device**, which is passed explicitly.
