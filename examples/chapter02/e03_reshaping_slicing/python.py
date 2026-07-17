"""Example 2.3 - Reshaping and Slicing (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    t = torch.arange(0, 12)
    m = t.reshape(3, 4)
    flat = m.flatten()          # back to a flat [12]
    piece = m[0:2, 1:3]         # rows 0-1, cols 1-2
    col = m.narrow(1, 0, 1)     # first column
    rows = m[[0, 2], :]         # gather rows 0 and 2
    return m, flat, piece, col, rows


def main():
    m, flat, piece, col, rows = build()
    print("reshaped =\n", m)
    print("flat     =\n", flat)
    print("piece    =\n", piece)
    print("first col=\n", col)
    print("rows 0,2 =\n", rows)


def test_matches_burn():
    m, flat, piece, col, rows = build()

    assert tuple(m.shape) == (3, 4)
    assert m.flatten().tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    assert tuple(flat.shape) == (12,)
    assert flat.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    assert tuple(piece.shape) == (2, 2)
    assert piece.flatten().tolist() == [1, 2, 5, 6]

    assert tuple(col.shape) == (3, 1)
    assert col.flatten().tolist() == [0, 4, 8]

    assert tuple(rows.shape) == (2, 4)
    assert rows.flatten().tolist() == [0, 1, 2, 3, 8, 9, 10, 11]


if __name__ == "__main__":
    main()
