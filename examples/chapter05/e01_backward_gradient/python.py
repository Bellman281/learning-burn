"""Example 5.1 - Backward Gradient (PyTorch reference + parity test).

Burn's autodiff vs PyTorch's autograd on the same function: f(x)=sum(x^2),
df/dx = 2x. Both engines should produce identical gradients.

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    f = (x * x).sum()   # 14
    f.backward()
    return f.item(), x.grad.tolist()   # grad = 2x = [2, 4, 6]


def main():
    f, dx = build()
    print("f  =", f)
    print("dx =", dx)


def test_matches_burn():
    f, dx = build()
    assert abs(f - 14.0) < 1e-6
    assert dx == [2.0, 4.0, 6.0]


if __name__ == "__main__":
    main()
