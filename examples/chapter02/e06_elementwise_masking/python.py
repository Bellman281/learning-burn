"""Example 2.6 - Element-wise Masking (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    x = torch.tensor([-1.0, 0.0, 2.0])
    y = torch.log1p(torch.exp(x))   # softplus
    c = torch.clamp(x, 0.0, 1.0)
    mask = x > 0                     # [False, False, True]
    picked = torch.where(mask, 9.0, x)
    return y, c, picked


def main():
    y, c, picked = build()
    print("softplus =", y)
    print("clamped  =", c)
    print("picked   =", picked)


def test_matches_burn():
    y, c, picked = build()

    expected = [0.31326166, 0.69314718, 2.1269281]
    for got, want in zip(y.tolist(), expected):
        assert abs(got - want) < 1e-4, f"y {got} vs {want}"

    assert c.tolist() == [0.0, 0.0, 1.0]
    assert picked.tolist() == [-1.0, 0.0, 9.0]


if __name__ == "__main__":
    main()
