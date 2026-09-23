"""Example 18.3 - Checkpoints and Resuming (PyTorch reference + parity test).

Save model AND optimiser state after 8 epochs, "crash", resume in a fresh
process-like setting, and finish: the weights equal a straight 15-epoch run.

Run:   python main.py
Test:  pytest main.py
"""

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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


class SmallCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.gap(x).flatten(1)
        return self.head(x)

    def with_fixed_weights(self):
        with torch.no_grad():
            self.conv1.weight.copy_(torch.tensor(det(36, 9, 0.1)).reshape(4, 1, 3, 3))
            self.conv1.bias.zero_()
            self.conv2.weight.copy_(torch.tensor(det(288, 36, 0.3)).reshape(8, 4, 3, 3))
            self.conv2.bias.zero_()
            self.head.weight.copy_(torch.tensor(det(32, 8, 0.5)).reshape(4, 8))
            self.head.bias.zero_()
        return self


class Shapes(Dataset):
    def __init__(self, per_class, seed):
        rng = Lcg(seed)
        self.items = [(make_image(i % 4, rng), i % 4) for i in range(per_class * 4)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pixels, label = self.items[i]
        return torch.from_numpy(pixels).reshape(1, SIZE, SIZE), label


@dataclass
class SmallCnnConfig:
    channels1: int = 4
    channels2: int = 8
    classes: int = 4

    def init(self):
        m = SmallCnn()
        m.conv1 = nn.Conv2d(1, self.channels1, 3, padding=1)
        m.conv2 = nn.Conv2d(self.channels1, self.channels2, 3, padding=1)
        m.head = nn.Linear(self.channels2, self.classes)
        return m


@dataclass
class TrainingConfig:
    model: SmallCnnConfig = field(default_factory=SmallCnnConfig)
    optimizer: dict = field(default_factory=lambda: {"epsilon": 1e-8})
    num_epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 0.02


def epoch(model, loader, optim=None):
    """One pass. Trains if an optimiser is given. Returns (mean loss, accuracy %)."""
    total, correct, n = 0.0, 0, 0
    for images, targets in loader:
        with torch.set_grad_enabled(optim is not None):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()
        total += loss.item() * len(targets)
        correct += (logits.argmax(1) == targets).sum().item()
        n += len(targets)
    return total / n, 100.0 * correct / n


def loaders(cfg):
    return (DataLoader(Shapes(40, 42), batch_size=cfg.batch_size, shuffle=False),
            DataLoader(Shapes(20, 7), batch_size=cfg.batch_size, shuffle=False))


def fresh(cfg):
    model = SmallCnn().with_fixed_weights()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-8)
    return model, optim


def build():
    cfg = TrainingConfig()
    train, _ = loaders(cfg)

    straight, optim = fresh(cfg)
    for _ in range(15):
        epoch(straight, train, optim)

    model, optim = fresh(cfg)
    for _ in range(8):
        epoch(model, train, optim)
    path = os.path.join(tempfile.gettempdir(), "learning-burn-c18e3.pt")
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "epoch": 8}, path)

    resumed, optim = fresh(cfg)                      # a brand-new model and optimiser...
    ckpt = torch.load(path)
    resumed.load_state_dict(ckpt["model"])           # ...given back their saved state
    optim.load_state_dict(ckpt["optim"])
    for _ in range(ckpt["epoch"], 15):
        epoch(resumed, train, optim)

    diff = (straight.conv1.weight - resumed.conv1.weight).abs().max().item()
    return sorted(ckpt.keys()), diff


def main():
    keys, diff = build()
    print("checkpoint holds:", keys)
    print(f"15 epochs straight vs 8 + resume + 7: max |weight difference| = {diff:e}")


def test_matches_burn():
    keys, diff = build()
    assert keys == ["epoch", "model", "optim"]
    assert diff == 0.0


if __name__ == "__main__":
    main()
