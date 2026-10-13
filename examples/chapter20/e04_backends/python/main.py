"""Example 20.4 - Backends (PyTorch reference + parity test).

PyTorch's "backend" is a device string. The same model on every device this
machine has; on CPU-only machines that is one row. Answers must agree.

Run:   python main.py
Test:  pytest main.py
"""

import importlib.util
import json
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "..", "..", "chapter19", "weights")

# Chapter 19's PyTorch model and data generator.
_spec = importlib.util.spec_from_file_location("make_weights", os.path.join(WEIGHTS, "make_weights.py"))
mw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mw)


def load_model():
    from safetensors.torch import load_file
    model = mw.TorchCnn()
    model.load_state_dict(load_file(os.path.join(WEIGHTS, "small_cnn.safetensors")))
    return model.eval()


def images(n):
    rng = mw.Lcg(7)
    return torch.stack([torch.from_numpy(mw.make_image(i % 4, rng)).reshape(1, 12, 12)
                        for i in range(n)])


@torch.no_grad()
def run(device):
    model = load_model().to(device)
    x = images(256).to(device)
    logits = model(x).cpu()
    times = []
    for _ in range(30):
        t = time.perf_counter()
        model(x).cpu()
        times.append((time.perf_counter() - t) * 1e3)
    return logits, sorted(times)[15]


def build():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + \
              (["mps"] if torch.backends.mps.is_available() else [])
    ref, ms = run("cpu")
    rows = [("cpu", ms, 0.0)]
    for d in devices[1:]:
        out, ms = run(d)
        rows.append((d, ms, (out - ref).abs().max().item()))
    return rows


def main():
    print("the same forward pass, 256 images, on each device")
    print("device    ms / batch   max |diff vs cpu|")
    for name, ms, diff in build():
        print(f"{name:<8}  {ms:>10.3f}   {diff:e}")


def test_devices_agree():
    assert all(diff < 1e-4 for _, _, diff in build())


if __name__ == "__main__":
    main()
