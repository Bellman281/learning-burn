"""Example 19.2 - Loading safetensors (PyTorch reference + parity test).

safetensors stores named arrays and nothing else: no pickle, no code. The
same file Burn's SafetensorsStore reads.

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
    from safetensors.torch import load_file
    state = load_file(os.path.join(WEIGHTS, "small_cnn.safetensors"))
    print("classifier.weight shape in the file:", list(state["classifier.weight"].shape))
    model = mw.TorchCnn()
    model.load_state_dict(state)
    images, _ = mw.test_images()
    logits = model.eval()(images)
    return logits.argmax(1).tolist(), (logits - reference()).abs().max().item()


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
