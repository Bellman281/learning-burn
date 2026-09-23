"""Example 19.4 - Convert Once (PyTorch reference + parity test).

The PyTorch-side habit that matches Burn's: convert the checkpoint once into
a plain weights file (here safetensors) and serve from that.

Run:   python main.py
Test:  pytest main.py
"""

import importlib.util
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "..", "weights")

# The model and the data generator are defined once, in weights/make_weights.py.
_spec = importlib.util.spec_from_file_location("make_weights", os.path.join(WEIGHTS, "make_weights.py"))
mw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mw)


def reference():
    with open(os.path.join(WEIGHTS, "reference.json")) as f:
        return torch.tensor(json.load(f)["logits"])


@torch.no_grad()
def build():
    import tempfile
    from safetensors.torch import load_file, save_file
    model = mw.TorchCnn()
    model.load_state_dict(torch.load(os.path.join(WEIGHTS, "small_cnn.pt")))
    out = os.path.join(tempfile.gettempdir(), "learning-burn-c19e4-small_cnn.safetensors")
    save_file(model.state_dict(), out)                  # convert once
    served = mw.TorchCnn()
    served.load_state_dict(load_file(out))               # serve from then on
    images, _ = mw.test_images()
    a, b = model.eval()(images), served.eval()(images)
    assert torch.equal(a, b)
    return b.argmax(1).tolist(), (b - reference()).abs().max().item()


def main():
    predicted, diff = build()
    print("predictions         =", predicted)
    print(f"max |logits - ref|  = {diff:e}")


def test_matches_burn():
    predicted, diff = build()
    assert predicted == [0, 1, 2, 3, 0, 1, 2, 3]
    assert diff < 1e-5


if __name__ == "__main__":
    main()
