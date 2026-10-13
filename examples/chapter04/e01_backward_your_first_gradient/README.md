# Autodiff: Computing Your First Gradient

This example demonstrates the basics of Automatic Differentiation (autodiff) in PyTorch and Burn, verifying the machine's calculus using the function $f(x) = \sum x^2$ (where the derivative $\frac{\partial f}{\partial x} = 2x$) for $x = [1, 2, 3]$.

### Key Concepts & Differences
- **Tracking Gradients:** PyTorch uses `requires_grad=True` on tensor creation, while Burn uses the `.require_grad()` method.
- **Autodiff Devices:** In Burn 0.22, autodiff is a property of the device: `Device::flex().autodiff()` gives a device whose tensors can record gradients (the example lists the `"autodiff"` feature explicitly; Burn's default features already include it via `"optim"`). This is close to PyTorch's `torch.device`, plus one switch.
- **Gradient Retrieval:** In PyTorch, calling `.backward()` populates `x.grad` in-place. In Burn, `f.backward()` returns a `grads` container, from which the gradient is explicitly extracted via `x.grad(&grads)`.
- **Scalar Loss:** Reduce to a single value before calling backward (hence the `.sum()`). PyTorch refuses a non-scalar `backward()`; Burn accepts it and seeds the backward pass with ones, which silently treats the output as if it had been summed.
