# Disabling Gradient Tracking (Inference)

This example shows how to perform cheap, memory-efficient inference by disabling gradient tracking in both PyTorch and Burn.

### Key Concepts & Differences
- **No Global State Blocks:** Unlike PyTorch, which relies on a dynamic global context manager (`with torch.no_grad():`), Burn has no special state blocks to enter and leave.
- **Taking a Tensor off Autodiff (`.inner()`):** In Burn 0.22, autodiff is a property of the device (`Device::flex().autodiff()`), and each tensor carries it. Calling `.inner()` (an alias of `.without_autodiff()`) returns the same values with autodiff switched off, so later operations record no graph; `is_autodiff()` on the result is `false`.
- **Per-Tensor, Not Global:** Whether a tensor can record is a property of that tensor, not a global mode: `is_autodiff()` says whether it is on autodiff at all, and `is_tracked()` whether it actually takes part in a recorded graph (a tensor on an autodiff device that was never marked is autodiff but not tracked). It is not checked by the compiler: `Tensor<1>` is the same type with or without autodiff. For inference-only programs, simply create tensors on `Device::flex()` without `.autodiff()`.
- **`detach()` is different:** Burn's `detach()` cuts a tensor's history but keeps it on autodiff (a marked leaf stays marked), so later operations are still recorded. PyTorch's `x.detach()` in `main.py` corresponds to Burn's `.inner()`.
