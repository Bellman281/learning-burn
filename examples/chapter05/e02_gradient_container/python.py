"""Example 5.2 - Gradient Container (PyTorch reference + parity test).

PyTorch stores each gradient on the tensor as `.grad`; Burn returns them all in
a container. Same gradients either way: L = sum(a*b) -> dL/da = b, dL/db = a.

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = torch.tensor([4.0, 5.0], requires_grad=True)

    loss = (a * b).sum()   # 23
    loss.backward()

    return loss.item(), a.grad.tolist(), b.grad.tolist()


def main():
    loss, da, db = build()
    print("loss  =", loss)
    print("dL/da =", da)
    print("dL/db =", db)


def test_matches_burn():
    loss, da, db = build()
    assert abs(loss - 23.0) < 1e-6
    assert da == [4.0, 5.0]   # = b
    assert db == [2.0, 3.0]   # = a


if __name__ == "__main__":
    main()
