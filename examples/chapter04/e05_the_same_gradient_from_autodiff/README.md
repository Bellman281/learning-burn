# Autodiff Gradient Descent

This example shows how to replace manual calculus with Automatic Differentiation (autodiff) to calculate the Mean Squared Error (MSE) gradient for linear regression.

### Key Concepts & Differences
- **No Manual Formulas:** Instead of deriving the gradient by hand, a single `loss.backward()` call applies the chain rule across all the tensor operations that built the loss. (There is no training loop here: the example computes one gradient, at $w = [0, 0]$.)
- **Equivalent Outputs:** Starting at $w = [0, 0]$, both PyTorch and Burn compute the exact same loss ($41.0$) and gradient ($[-12.0, -35.0]$) as the manual formula from the previous example.
- **Implementation in Burn:** 
  - The tensors are created on an autodiff device, `Device::flex().autodiff()`.
  - The weight tensor $w$ is marked with `.require_grad()`.
  - The gradient is safely retrieved from the returned container using `w.grad(&grads)`.
