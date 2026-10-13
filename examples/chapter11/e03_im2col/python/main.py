"""Example 11.3 - im2col: a Convolution as One Matmul (PyTorch reference + parity test).

Unfold every 3x3 patch of the padded input into a column, reshape the kernels
into rows, and the whole convolution becomes one matrix product. Checked
against F.unfold and F.conv2d.

Run:   python main.py
Test:  pytest main.py
"""

import torch
import torch.nn.functional as F


class Lcg:
    """Tiny LCG, bit-identical to the Rust one."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF

    def small(self):
        return float(self.next() % 7) - 3.0   # an integer in -3..=3


def out_len(n, k, padding, stride, dilation):
    return (n + 2 * padding - dilation * (k - 1) - 1) // stride + 1


def im2col(x, ci, h, w, k, stride, padding):
    """One column per output position, one row per (channel, kernel row, kernel col)."""
    ho, wo = out_len(h, k, padding, stride, 1), out_len(w, k, padding, stride, 1)
    cols = [[0.0] * (ho * wo) for _ in range(ci * k * k)]
    for c in range(ci):
        for a in range(k):
            for e in range(k):
                row = (c * k + a) * k + e
                for i in range(ho):
                    for j in range(wo):
                        r, q = i * stride + a - padding, j * stride + e - padding
                        if 0 <= r < h and 0 <= q < w:
                            cols[row][i * wo + j] = x[(c * h + r) * w + q]
    return cols, ho, wo


def cost(c_in, c_out, k, ho, wo):
    """(columns shape, weight shape, multiply-adds, entries of the column matrix)."""
    rows, n_cols = c_in * k * k, ho * wo
    return [rows, n_cols], [c_out, rows], c_out * rows * n_cols, rows * n_cols


def build():
    rng = Lcg(11)
    x = [rng.small() for _ in range(50)]   # [1, 2, 5, 5], same as Example 11.2
    w = [rng.small() for _ in range(54)]   # [3, 2, 3, 3]
    bias = [1.0, -1.0, 0.0]

    cols, ho, wo = im2col(x, 2, 5, 5, 3, 1, 1)
    cols_t = torch.tensor(cols)                          # [18, 25]
    w_mat = torch.tensor(w).reshape(3, 18)               # each kernel flattened to a row
    y = w_mat @ cols_t + torch.tensor(bias).reshape(3, 1)  # [3, 18] @ [18, 25] -> [3, 25]
    y = y.reshape(1, 3, ho, wo)

    xt = torch.tensor(x).reshape(1, 2, 5, 5)
    unfolded = F.unfold(xt, 3, padding=1)                # [1, 18, 25]
    conv = F.conv2d(xt, torch.tensor(w).reshape(3, 2, 3, 3), torch.tensor(bias), padding=1)
    cols_diff = (unfolded[0] - cols_t).abs().max().item()
    y_diff = (y - conv).abs().max().item()
    first_patch = [cols[r][0] for r in range(18)]
    return (list(cols_t.shape), list(w_mat.shape), list(y.shape), first_patch,
            cols_diff, y_diff, y.flatten().tolist(), cost(2, 3, 3, ho, wo),
            cost(64, 64, 3, 56, 56))


def main():
    (cols_shape, w_shape, y_shape, first, cols_diff, y_diff, y, toy, big) = build()
    print(f"columns  {cols_shape}   weights {w_shape}   y {y_shape}")
    print("first column (the padded patch under output (0, 0)):")
    print("   channel 0:", first[:9])
    print("   channel 1:", first[9:])
    print(f"max|im2col - F.unfold|   = {cols_diff}")
    print(f"max|matmul - F.conv2d|   = {y_diff}")
    print("y[0, 0] (padding 1) =")
    for r in range(5):
        print("   " + "".join(f"{y[r * 5 + c]:6.1f}" for c in range(5)))
    for name, (cs, ws, macs, entries) in [("this example", toy), ("64->64, 3x3, 56x56", big)]:
        print(f"{name:<20} columns {cs}  weights {ws}  multiply-adds {macs}  column entries {entries}")


def test_matches_burn():
    (cols_shape, w_shape, y_shape, first, cols_diff, y_diff, y, toy, big) = build()
    assert cols_shape == [18, 25] and w_shape == [3, 18] and y_shape == [1, 3, 5, 5]
    assert first == [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -3.0, -1.0,
                     0.0, 0.0, 0.0, 0.0, -1.0, -3.0, 0.0, 3.0, 2.0]
    assert cols_diff == 0.0 and y_diff == 0.0
    # The inner 3x3 of channel 0 is Example 11.2's stride-1 output.
    assert [y[r * 5 + c] for r in range(1, 4) for c in range(1, 4)] == \
        [-1.0, -5.0, 5.0, 1.0, -35.0, 20.0, 10.0, -12.0, 20.0]
    assert toy == ([18, 25], [3, 18], 1350, 450)
    assert big == ([576, 3136], [64, 576], 115605504, 1806336)


if __name__ == "__main__":
    main()
