"""Example 11.7 - A Tiny CNN Trained by Hand (PyTorch reference + parity test).

conv 1 -> 4 channels (3x3, padding 1) -> relu -> 2x2 max-pool -> linear 144 -> 4,
trained with plain SGD on the 12x12 "shapes" images. Same data, same order,
same fixed initial weights as the Rust example, which computes every gradient
by hand. Here autograd computes them, and the per-epoch losses must match.

Run:   python main.py
Test:  pytest main.py
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

SIZE = 12
LR = 0.3
EPOCHS = 10
BATCH = 16


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
    """class 0 = horizontal bar, 1 = vertical bar, 2 = diagonal, 3 = plus sign"""
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


def shapes(per_class, seed):
    rng = Lcg(seed)
    images = np.stack([make_image(i % 4, rng) for i in range(per_class * 4)])
    labels = [i % 4 for i in range(per_class * 4)]
    return torch.from_numpy(images).reshape(-1, 1, SIZE, SIZE), torch.tensor(labels)


def det(n, fan_in, phase):
    return torch.tensor([math.sin(i * 2.3 + phase) / math.sqrt(fan_in) for i in range(n)],
                        dtype=torch.float32)


def init():
    """Fixed weights. The linear weight is [in, out] = [144, 4], as in Chapter 8."""
    return [det(36, 9, 0.1).reshape(4, 1, 3, 3), torch.zeros(4, dtype=torch.float32),
            det(576, 144, 0.5).reshape(144, 4), torch.zeros(4, dtype=torch.float32)]


def forward(params, x):
    cw, cb, lw, lb = params
    h = F.max_pool2d(F.relu(F.conv2d(x, cw, cb, padding=1)), 2)   # [B, 4, 6, 6]
    return h.flatten(1) @ lw + lb                                   # [B, 4]


def step_grads(params, x, y):
    params = [p.clone().requires_grad_() for p in params]
    loss = F.cross_entropy(forward(params, x), y)
    loss.backward()
    return loss.item(), [p.grad for p in params]


def build():
    xs, ys = shapes(100, 42)    # 400 training images
    xt, yt = shapes(20, 7)      # 80 test images
    params = init()
    first_loss, first_grads = step_grads(params, xs[:BATCH], ys[:BATCH])
    history = []
    for _ in range(EPOCHS):
        total = 0.0
        for s in range(0, len(ys), BATCH):
            loss, grads = step_grads(params, xs[s:s + BATCH], ys[s:s + BATCH])
            params = [p - LR * g for p, g in zip(params, grads)]   # plain SGD
            total += loss
        history.append(total / (len(ys) // BATCH))
    with torch.no_grad():
        correct = (forward(params, xt).argmax(1) == yt).sum().item()
    return first_loss, first_grads, history, correct


def main():
    first_loss, (gcw, gcb, glw, glb), history, correct = build()
    fmt = lambda t: "[" + ", ".join(f"{v:.6f}" for v in t.tolist()) + "]"
    print(f"step 1: loss {first_loss:.6f}")
    print(f"  dL/d conv bias       = {fmt(gcb)}")
    print(f"  dL/d conv w[0, 0, 0] = {fmt(gcw[0, 0, 0])}")
    print(f"  dL/d linear bias     = {fmt(glb)}")
    for e, loss in enumerate(history, 1):
        print(f"epoch {e}  mean loss {loss:.6f}")
    print(f"test accuracy = {correct}/80")


# Mean loss per epoch from the Rust example, whose gradients are all computed by hand.
EXPECTED_HISTORY = [1.309250, 1.064798, 0.786106, 0.534784, 0.371480,
                    0.264373, 0.192489, 0.144322, 0.112260, 0.090219]


def test_matches_burn():
    first_loss, (gcw, gcb, glw, glb), history, correct = build()
    assert abs(first_loss - 1.372874) < 1e-6
    assert all(abs(a - b) < 1e-6 for a, b in
               zip(gcb.tolist(), [-0.016677, -0.011590, -0.036865, -0.015081]))
    assert all(abs(a - b) < 1e-4 for a, b in zip(history, EXPECTED_HISTORY))
    assert correct == 79


if __name__ == "__main__":
    main()
