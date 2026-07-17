"""Example 2.4 - Reductions (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    total = x.sum()                          # 10.
    col_sum = x.sum(dim=0, keepdim=True)     # [[4., 6.]]
    row_mean = x.mean(dim=1, keepdim=True)   # [[1.5], [3.5]]
    m, idx = x.max(dim=1)                    # values [2., 4.], indices [1, 1]
    return total, col_sum, row_mean, m, idx


def main():
    total, col_sum, row_mean, m, idx = build()
    print("total     =", total)
    print("col_sum   =", col_sum)
    print("row_mean  =", row_mean)
    print("max vals  =", m)
    print("max idx   =", idx)


def test_matches_burn():
    total, col_sum, row_mean, m, idx = build()

    assert total.item() == 10.0
    assert col_sum.flatten().tolist() == [4.0, 6.0]
    assert row_mean.flatten().tolist() == [1.5, 3.5]

    # torch's max(dim) drops the reduced dim ([2]); Burn keeps it ([2,1]).
    # Values and indices match once flattened.
    assert m.flatten().tolist() == [2.0, 4.0]
    assert idx.flatten().tolist() == [1, 1]


if __name__ == "__main__":
    main()
