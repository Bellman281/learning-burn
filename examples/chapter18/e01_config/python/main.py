"""Example 18.1 - Configs (PyTorch reference + parity test).

PyTorch has no Config type; a dataclass plus json is the usual stand-in.
Same fields, same defaults, same parameter counts as rust/src/main.rs.

Run:   python main.py
Test:  pytest main.py
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field

import torch.nn as nn


@dataclass
class SmallCnnConfig:
    channels1: int = 4
    channels2: int = 8
    classes: int = 4

    def init(self):
        return nn.ModuleDict({
            "conv1": nn.Conv2d(1, self.channels1, 3, padding=1),
            "conv2": nn.Conv2d(self.channels1, self.channels2, 3, padding=1),
            "head": nn.Linear(self.channels2, self.classes),
        })


@dataclass
class TrainingConfig:
    model: SmallCnnConfig = field(default_factory=SmallCnnConfig)
    optimizer: dict = field(default_factory=lambda: {"epsilon": 1e-8})
    num_epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 0.02


def params(m):
    return sum(p.numel() for p in m.parameters())


def build():
    config = TrainingConfig()
    path = os.path.join(tempfile.gettempdir(), "learning-burn-c18e1-config.json")
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    with open(path) as f:
        raw = json.load(f)
    loaded = TrainingConfig(model=SmallCnnConfig(**raw.pop("model")), **raw)
    wider = SmallCnnConfig(**{**asdict(loaded.model), "channels2": 16})
    return open(path).read(), params(loaded.model.init()), params(wider.init()), loaded.num_epochs


def main():
    text, p, wider, epochs = build()
    print(text)
    print("epochs read back       =", epochs)
    print("default model params   =", p)
    print("channels2 = 16, params =", wider)


def test_matches_burn():
    _, p, wider, epochs = build()
    assert (p, wider, epochs) == (372, 700, 15)


if __name__ == "__main__":
    main()
