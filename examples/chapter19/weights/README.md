# Chapter 19 weights

A CNN trained in PyTorch and exported three ways, plus PyTorch's own logits for
8 test images. `make_weights.py` regenerates everything deterministically
(needs torch, safetensors, onnx).

| File | Used by |
|---|---|
| `small_cnn.pt` | Examples 19.1, 19.4 |
| `small_cnn.safetensors` | Example 19.2, and the Chapter 20 server |
| `small_cnn.onnx` | Example 19.3 (`build.rs`) |
| `reference.json` | every Chapter 19 test: the logits Burn must reproduce |
