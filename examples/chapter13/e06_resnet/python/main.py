"""Example 13.6 - Residual Blocks and a Small ResNet (PyTorch reference + parity test).

Conv-BN-ReLU-Conv-BN plus a skip connection (a 1x1 projection when the shape
changes), stacked into a small ResNet with fixed deterministic weights. The
logits must match Burn's in eval mode (running statistics) and in train mode
(batch statistics).

Run:   python main.py
Test:  pytest main.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Filler:
    """Every tensor gets its own phase, in the same order as main.rs."""

    def __init__(self):
        self.phase = 0.0

    def wave(self, n, step, scale, offset):
        p = self.phase
        self.phase += 1.0
        return torch.tensor([offset + scale * math.sin(i * step + p) for i in range(n)],
                            dtype=torch.float32)

    def tensor(self, shape, fan_in):
        return self.wave(math.prod(shape), 2.3, 1 / math.sqrt(fan_in), 0.0).reshape(shape)

    def vector(self, n, scale, offset):
        return self.wave(n, 1.7, scale, offset)


class ConvBn(nn.Module):
    def __init__(self, c_in, c_out, k, stride):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return self.bn(self.conv(x))

    def fill(self, f):
        c_out, c_in, k, _ = self.conv.weight.shape
        self.conv.weight.copy_(f.tensor([c_out, c_in, k, k], c_in * k * k))
        self.bn.weight.copy_(f.vector(c_out, 0.1, 1.0))
        self.bn.bias.copy_(f.vector(c_out, 0.1, 0.0))
        self.bn.running_mean.copy_(f.vector(c_out, 0.1, 0.0))
        self.bn.running_var.copy_(f.vector(c_out, 0.25, 1.0))


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.a = ConvBn(c_in, c_out, 3, stride)
        self.b = ConvBn(c_out, c_out, 3, 1)
        self.project = ConvBn(c_in, c_out, 1, stride) if (c_in != c_out or stride != 1) else None

    def forward(self, x):
        skip = x if self.project is None else self.project(x)
        return F.relu(self.b(F.relu(self.a(x))) + skip)

    def fill(self, f):
        self.a.fill(f)
        self.b.fill(f)
        if self.project is not None:
            self.project.fill(f)


class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBn(1, 8, 3, 1)
        self.block1 = ResidualBlock(8, 8, 1)
        self.block2 = ResidualBlock(8, 16, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = self.block2(self.block1(x))
        return self.head(self.gap(x).flatten(1))

    @torch.no_grad()
    def with_fixed_weights(self):
        f = Filler()
        self.stem.fill(f)
        self.block1.fill(f)
        self.block2.fill(f)
        self.head.weight.copy_(f.tensor([4, 16], 16))
        self.head.bias.copy_(f.vector(4, 0.1, 0.0))
        return self


def build():
    rng = Lcg(42)
    x = torch.from_numpy(np.stack([make_image(c, rng) for c in range(4)])).reshape(4, 1, SIZE, SIZE)
    model = SmallResNet().with_fixed_weights()
    with torch.no_grad():
        model.eval()
        eval_logits = model(x).flatten().tolist()
        model.train()
        train_logits = model(x).flatten().tolist()
    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for name, b in model.named_buffers() if "running" in name)
    return eval_logits, train_logits, params, buffers


EXPECTED_EVAL = [-0.030468, 0.076172, -0.018825, -0.123167, -0.024731, 0.077487, -0.022925, -0.129586, -0.037276, 0.075673, -0.012638, -0.114969, -0.071079, 0.035744, -0.028530, -0.094820]
EXPECTED_TRAIN = [-0.072590, 0.023997, -0.041640, -0.099389, -0.041734, -0.097797, -0.224083, -0.204665, -0.067853, 0.000614, -0.075479, -0.118123, -0.091215, -0.111750, -0.191967, -0.150741]


def main():
    eval_logits, train_logits, params, buffers = build()
    for c in range(4):
        print(f"eval  logits, class {c} image: [" + ", ".join(f"{v:.4f}" for v in eval_logits[c * 4:c * 4 + 4]) + "]")
    for c in range(4):
        print(f"train logits, class {c} image: [" + ", ".join(f"{v:.4f}" for v in train_logits[c * 4:c * 4 + 4]) + "]")
    print(f"parameters() = {params}, running-stat buffers = {buffers}")


def test_matches_burn():
    eval_logits, train_logits, params, buffers = build()
    assert all(abs(a - b) < 1e-5 for a, b in zip(eval_logits, EXPECTED_EVAL))
    assert all(abs(a - b) < 1e-5 for a, b in zip(train_logits, EXPECTED_TRAIN))
    assert (params, buffers) == (5020, 144)


if __name__ == "__main__":
    main()
