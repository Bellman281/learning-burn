# Backend-Agnostic Inference (Example 21.2)

A demonstration showing how **Burn** decouples model logic from execution hardware, allowing the exact same code to run on CPU, GPU, or embedded microcontrollers.

## The Core Concept
In PyTorch, the device (`torch.device("cuda")`, `"cpu"`) is a runtime value. Since Burn 0.22 the same is true in Burn: the model and the inference function take a `&Device`, and the backend behind it is chosen when the program runs (`Device::flex()`, `Device::wgpu(...)`, `Device::cuda(0)`, ...). The tensor type carries only the rank and kind (`Tensor<2>`), never the backend.

## What is decided when
- **Run time:** which device a call uses. `run_inference(&Device::flex())` and `run_inference(&gpu)` are the same compiled function.
- **Compile time:** which backends exist in the binary at all. Each backend sits behind a Cargo feature (`flex`, `wgpu`, `cuda`, ...); only the enabled ones are compiled in. On a microcontroller you enable `flex` alone, with Burn's default features off, so the firmware carries one pure-Rust CPU backend and nothing else.
- **Device mismatches** (a tensor on one backend meeting a tensor on another) are therefore caught at run time, as in PyTorch, not by the compiler.
