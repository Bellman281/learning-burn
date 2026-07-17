"""Example 2.2 - Broadcasting (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    m = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bias = torch.tensor([100.0, 200.0, 300.0])
    # PyTorch inserts the missing axis for you; Burn asks for unsqueeze().
    return m + bias


def main():
    print(build())


def test_matches_burn():
    out = build()
    assert tuple(out.shape) == (2, 3)
    assert out.tolist() == [[101.0, 202.0, 303.0], [104.0, 205.0, 306.0]]


if __name__ == "__main__":
    main()
