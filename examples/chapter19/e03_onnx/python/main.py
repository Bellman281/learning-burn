"""Example 19.3 - Running the ONNX File (onnxruntime reference + parity test).

The ONNX graph run by onnxruntime -- a third, independent engine. Burn's
generated code must agree with it and with PyTorch.

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


def build():
    import onnx
    import onnxruntime as ort
    path = os.path.join(WEIGHTS, "small_cnn.onnx")
    print("ONNX operators:", [n.op_type for n in onnx.load(path).graph.node])
    session = ort.InferenceSession(path)
    images, _ = mw.test_images()
    logits = torch.tensor(session.run(["logits"], {"images": images.numpy()})[0])
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
