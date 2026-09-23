"""Regenerates this folder: a CNN trained in PyTorch, exported three ways.

  small_cnn.pt           torch.save(model.state_dict())      -> Example 19.1
  small_cnn.safetensors  safetensors.torch.save_file(...)    -> Example 19.2
  small_cnn.onnx         torch.onnx.export(...)              -> Example 19.3
  reference.json         PyTorch's logits for the first 8 test images, the
                         numbers every Rust example must reproduce.

The model is Chapter 15's CNN written the way PyTorch code usually is: an
nn.Sequential called "features" and a Linear called "classifier". Its
parameter names therefore do NOT match the Burn struct's field names, which
is the point of Example 19.2.

Run:  python make_weights.py      (needs torch, safetensors, onnx)
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))

SIZE = 12


class Lcg:
    """Tiny LCG, bit-identical to the Rust one."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF

    def below(self, n):
        return self.next() % n


def make_image(cls, rng):
    img = np.zeros(SIZE * SIZE, dtype=np.float32)
    if cls == 0:
        r, c = rng.below(12), rng.below(7)
        for j in range(6):
            img[r * SIZE + c + j] = 1.0
    elif cls == 1:
        c, r = rng.below(12), rng.below(7)
        for i in range(6):
            img[(r + i) * SIZE + c] = 1.0
    elif cls == 2:
        r, c = rng.below(7), rng.below(7)
        for i in range(6):
            img[(r + i) * SIZE + c + i] = 1.0
    else:
        r, c = 2 + rng.below(8), 2 + rng.below(8)
        for d in range(5):
            img[r * SIZE + c + d - 2] = 1.0
            img[(r + d - 2) * SIZE + c] = 1.0
    for p in range(SIZE * SIZE):
        img[p] += np.float32(rng.next()) / np.float32(32768.0) * np.float32(0.25)
    return img


def det(n, fan_in, phase):
    return [math.sin(i * 2.3 + phase) / math.sqrt(fan_in) for i in range(n)]


class TorchCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),   # features.0
            nn.ReLU(),                       # features.1  (no weights)
            nn.MaxPool2d(2),                 # features.2  (no weights)
            nn.Conv2d(4, 8, 3, padding=1),   # features.3
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(8, 4)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


class Shapes(Dataset):
    def __init__(self, per_class, seed):
        rng = Lcg(seed)
        self.items = [(make_image(i % 4, rng), i % 4) for i in range(per_class * 4)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pixels, label = self.items[i]
        return torch.from_numpy(pixels).reshape(1, SIZE, SIZE), label


def test_images(n=8):
    test = Shapes(20, 7)
    return torch.stack([test[i][0] for i in range(n)]), [test[i][1] for i in range(n)]


def main():
    torch.manual_seed(2)
    model = TorchCnn()
    optim = torch.optim.Adam(model.parameters(), lr=0.02)
    loader = DataLoader(Shapes(40, 42), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(2))
    for _ in range(15):
        for images, targets in loader:
            loss = F.cross_entropy(model(images), targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
    model.eval()

    torch.save(model.state_dict(), os.path.join(HERE, "small_cnn.pt"))

    from safetensors.torch import save_file
    save_file(model.state_dict(), os.path.join(HERE, "small_cnn.safetensors"))

    torch.onnx.export(model, torch.zeros(1, 1, SIZE, SIZE), os.path.join(HERE, "small_cnn.onnx"),
                      dynamo=False, opset_version=16,
                      input_names=["images"], output_names=["logits"],
                      dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}})

    images, labels = test_images()
    with torch.no_grad():
        logits = model(images)
    ref = {"labels": labels, "logits": [[float(v) for v in row] for row in logits.tolist()],
           "parameter_names": list(model.state_dict().keys())}
    with open(os.path.join(HERE, "reference.json"), "w") as f:
        json.dump(ref, f, indent=1)
    print("predicted", logits.argmax(1).tolist(), "labels", labels)
    for name in ("small_cnn.pt", "small_cnn.safetensors", "small_cnn.onnx"):
        print(f"{name:<22} {os.path.getsize(os.path.join(HERE, name)):>6} bytes")


if __name__ == "__main__":
    main()
