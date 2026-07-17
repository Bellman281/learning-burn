"""Example 2.1 - Element-wise Arithmetic (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])
    s = a + b       # element-wise add
    p = a * b       # element-wise multiply, NOT matmul
    sc = a * 2.0
    neg = -a
    return s, p, sc, neg


def main():
    s, p, sc, neg = build()
    print("sum    =", s)
    print("prod   =", p)
    print("scaled =", sc)
    print("neg    =", neg)


def test_matches_burn():
    s, p, sc, neg = build()
    assert s.tolist() == [11.0, 22.0, 33.0]
    assert p.tolist() == [10.0, 40.0, 90.0]
    assert sc.tolist() == [2.0, 4.0, 6.0]
    assert neg.tolist() == [-1.0, -2.0, -3.0]


if __name__ == "__main__":
    main()
