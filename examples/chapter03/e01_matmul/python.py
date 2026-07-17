"""Example 3.1 - Matrix Multiplication (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # [2, 3]
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # [3, 2]
    return a @ b  # [2, 2]  (element-wise * would NOT be matmul)


def main():
    print(build())


def test_matches_burn():
    c = build()
    assert tuple(c.shape) == (2, 2)
    assert c.flatten().tolist() == [4.0, 5.0, 10.0, 11.0]


if __name__ == "__main__":
    main()
