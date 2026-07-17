"""Example 5.3 - Inner / Inference (PyTorch reference + parity test).

Burn drops autodiff with `.inner()`; PyTorch's closest analogue is a
`torch.no_grad()` block (or `.detach()`). Same result either way.

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    x = torch.tensor([1.0, 2.0, 3.0])
    with torch.no_grad():          # skip graph tracking, like Burn's inner()
        y = x + 5.0
    return y.tolist()              # [6, 7, 8]


def main():
    print(build())


def test_matches_burn():
    assert build() == [6.0, 7.0, 8.0]


if __name__ == "__main__":
    main()
