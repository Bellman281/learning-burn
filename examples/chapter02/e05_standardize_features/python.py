"""Example 2.5 - Standardize Features (PyTorch reference + parity test).

Both frameworks use UNBIASED variance (/(n-1)) by default, so the results match.

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean = x.mean(dim=0, keepdim=True)        # [[3., 4.]]
    std = x.var(dim=0, keepdim=True).sqrt()   # [[2., 2.]]  (unbiased, /(n-1))
    z = (x - mean) / std
    return mean, std, z


def main():
    mean, std, z = build()
    print("mean =", mean)
    print("std  =", std)
    print("z    =\n", z)


def _approx(got, want, tol=1e-5):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert abs(g - w) < tol, f"got {g}, want {w}"


def test_matches_burn():
    mean, std, z = build()
    _approx(mean.flatten().tolist(), [3.0, 4.0])
    _approx(std.flatten().tolist(), [2.0, 2.0])
    _approx(z.flatten().tolist(), [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0])


if __name__ == "__main__":
    main()
