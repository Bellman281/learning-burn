"""Example 13.1 - Burn's Conv2d Layer, Next to Chapter 11's Function (PyTorch reference + parity test).

The vertical-edge kernel run twice: through F.conv2d with the weight passed in
(Chapter 11's way) and through an nn.Conv2d layer that owns the same weight.
Then the default init bound and the output-size rule for padding, stride and
dilation.

Run:   python main.py
Test:  pytest main.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def out_len(n, k, padding, stride, dilation):
    return (n + 2 * padding - dilation * (k - 1) - 1) // stride + 1


def build():
    image = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]] * 5).reshape(1, 1, 5, 5)
    kernel = torch.tensor([[-1.0, 0.0, 1.0]] * 3).reshape(1, 1, 3, 3)
    by_function = F.conv2d(image, kernel)

    layer = nn.Conv2d(1, 1, 3, bias=False)
    with torch.no_grad():
        layer.weight.copy_(kernel)            # replace the random init
    by_layer = layer(image).detach()
    print("edge map shape =", list(by_layer.shape))   # [1, 1, 3, 3]

    fresh = nn.Conv2d(1, 8, 3)                 # default init
    within = bool((fresh.weight.abs() <= 1 / math.sqrt(9)).all())

    x = torch.zeros(1, 1, 12, 12)
    cases = [("valid, stride 1", 0, 1, 1), ("padding 1", 1, 1, 1),
             ("padding 1, stride 2", 1, 2, 1), ("dilation 2", 0, 1, 2)]
    shapes = []
    for name, p, s, d in cases:
        conv = nn.Conv2d(1, 8, 3, padding=p, stride=s, dilation=d)
        weight_dims = list(conv.weight.shape)
        shapes.append((name, out_len(12, 3, p, s, d), conv(x).shape[2]))
    return (by_function.flatten().tolist(), by_layer.flatten().tolist(), within,
            shapes, weight_dims)


def main():
    by_function, by_layer, within, shapes, weight_dims = build()
    print("conv2d function =", by_function)
    print("Conv2d layer    =", by_layer)
    print("default init: all 72 weights within +-1/sqrt(9):", within)
    print("conv weight     =", weight_dims, " (out, in, kh, kw)")
    for name, predicted, actual in shapes:
        print(f"12x12, 3x3, {name:<20} -> rule {predicted:>2}, Conv2d {actual:>2}")


def test_matches_burn():
    by_function, by_layer, within, shapes, weight_dims = build()
    edges = [3.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0, 3.0, 0.0]
    assert by_function == edges and by_layer == edges
    assert within
    assert weight_dims == [8, 1, 3, 3]
    assert [s[2] for s in shapes] == [10, 12, 6, 8]
    assert all(s[1] == s[2] for s in shapes)


if __name__ == "__main__":
    main()
