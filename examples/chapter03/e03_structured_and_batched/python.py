"""Example 3.3 - Structured Matrices and Batched Matmul (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    m = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    eye = torch.eye(3)

    lo = torch.tril(m)       # lower-triangular part
    up = torch.triu(m)       # upper-triangular part
    tr = torch.trace(m)      # sum of the diagonal

    # Batched matmul: 8 matrices [2,3] times 8 matrices [3,4] -> [8, 2, 4]
    ba = torch.zeros(8, 2, 3)
    bb = torch.zeros(8, 3, 4)
    bc = ba @ bb
    return eye, lo, up, tr, bc


def main():
    eye, lo, up, tr, bc = build()
    print("identity =\n", eye)
    print("lo       =", lo)
    print("up       =", up)
    print("trace    =", tr)
    print("batched dims =", tuple(bc.shape))


def test_matches_burn():
    eye, lo, up, tr, bc = build()

    assert eye.flatten().tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert lo.flatten().tolist() == [1.0, 0.0, 3.0, 4.0]
    assert up.flatten().tolist() == [1.0, 2.0, 0.0, 4.0]
    assert tr.item() == 5.0

    assert tuple(bc.shape) == (8, 2, 4)
    assert bc.flatten().tolist() == [0.0] * 64


if __name__ == "__main__":
    main()
