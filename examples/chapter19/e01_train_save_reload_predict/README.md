# Training, Saving, and Inference Round Trip (Example 19.1)

A complete end-to-end pipeline demonstrating how to train a model, serialize its weights to disk, and reload them for safe, lightweight inference in **Burn** and **PyTorch**.

## The Core Concept
Modern deep learning workflows separate training from deployment. In Burn 0.22 the division is a property of the **device**, not of the model's type:
- **Training Mode:** The model is built on `Device::flex().autodiff()`, so its tensors can record the computational graph for gradient updates.
- **Inference Mode:** The same `Model` struct is loaded on plain `Device::flex()` (no autodiff), so nothing is recorded.

Dropout and batch normalization only behave as in training on an autodiff device, so the inference model needs no `.eval()` call; `model.valid()` is the explicit switch when you need one.

## Key Takeaways
- **No More `.eval()` Bugs:** In PyTorch, forgetting `.eval()` keeps dropout and batch normalization active during inference, leading to silent bugs. In Burn, a model loaded on a device without autodiff has no training-only behavior to switch off.
- **Ownership on Save:** The `.save_file()` function in Burn consumes the model by value. To continue using or testing a model after saving, you must clone it before serialization.
- **Burnpack:** `model.save_file(path)` writes the weights as a burnpack (`.bpk`) file; `Model::new(&device).try_load_file(path)` maps them back onto a freshly initialized model (`load_file` is the panicking variant).
