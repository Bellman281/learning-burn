"""Example 3.2 - Transpose, Matrix-Vector, Outer Product (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    m = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    v = torch.tensor([1.0, 1.0])

    mt = m.t()               # transpose
    mv = m @ v               # matrix * vector -> [2]
    d = torch.dot(v, v)      # scalar dot product
    op = torch.outer(v, v)   # outer product -> [2, 2]
    return mt, mv, d, op


def main():
    mt, mv, d, op = build()
    print("transpose =\n", mt)
    print("matvec    =", mv)
    print("dot       =", d)
    print("outer     =\n", op)


def test_matches_burn():
    mt, mv, d, op = build()

    assert tuple(mt.shape) == (2, 2)
    assert mt.flatten().tolist() == [1.0, 3.0, 2.0, 4.0]

    assert mv.tolist() == [3.0, 7.0]

    assert d.item() == 2.0

    assert tuple(op.shape) == (2, 2)
    assert op.flatten().tolist() == [1.0, 1.0, 1.0, 1.0]


if __name__ == "__main__":
    main()
