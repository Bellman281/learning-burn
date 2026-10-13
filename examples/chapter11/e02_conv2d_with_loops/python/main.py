"""Example 11.2 - 2D Convolution with Loops (PyTorch reference + parity test).

A multi-channel 2-D cross-correlation with stride, zero padding and dilation,
written as seven nested loops over flat lists, then checked against
F.conv2d. The data are small integers, so every sum is exact.

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


def conv2d_loops(x, x_shape, w, w_shape, bias, stride, padding, dilation):
    n, ci, h, wd = x_shape
    co, _, kh, kw = w_shape
    ho = out_len(h, kh, padding, stride, dilation)
    wo = out_len(wd, kw, padding, stride, dilation)
    y = [0.0] * (n * co * ho * wo)
    for b in range(n):
        for o in range(co):
            for i in range(ho):
                for j in range(wo):
                    acc = bias[o]
                    for c in range(ci):
                        for a in range(kh):
                            for e in range(kw):
                                r = i * stride + a * dilation - padding
                                q = j * stride + e * dilation - padding
                                if r < 0 or q < 0 or r >= h or q >= wd:
                                    continue   # a padding zero
                                acc += (w[((o * ci + c) * kh + a) * kw + e]
                                        * x[((b * ci + c) * h + r) * wd + q])
                    y[((b * co + o) * ho + i) * wo + j] = acc
    return y, [n, co, ho, wo]


CASES = [("stride 1", 1, 0, 1), ("padding 1", 1, 1, 1),
         ("stride 2, padding 1", 2, 1, 1), ("dilation 2, padding 2", 1, 2, 2)]


def build():
    rng = Lcg(11)
    x_shape, w_shape = [1, 2, 5, 5], [3, 2, 3, 3]
    x = [rng.small() for _ in range(50)]
    w = [rng.small() for _ in range(54)]
    bias = [1.0, -1.0, 0.0]

    xt = torch.tensor(x).reshape(x_shape)
    wt = torch.tensor(w).reshape(w_shape)
    bt = torch.tensor(bias)
    results = []
    for name, s, p, d in CASES:
        y, shape = conv2d_loops(x, x_shape, w, w_shape, bias, s, p, d)
        yt = F.conv2d(xt, wt, bt, stride=s, padding=p, dilation=d)
        diff = max(abs(a - b) for a, b in zip(y, yt.flatten().tolist()))
        rule = out_len(5, 3, p, s, d)
        results.append((name, rule, shape, list(yt.shape), diff, y))
    params = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3] + w_shape[0]
    return x, w, results, params


def grid(v, rows, cols):
    for r in range(rows):
        print("   " + "".join(f"{v[r * cols + c]:6.1f}" for c in range(cols)))


def main():
    x, w, results, params = build()
    for c in range(2):
        print(f"x[0, {c}] =")
        grid(x[c * 25:], 5, 5)
    for c in range(2):
        print(f"w[0, {c}] =")
        grid(w[c * 9:], 3, 3)
    print("bias    = [1.0, -1.0, 0.0]")
    print(f"params  = {params}  (3*2*3*3 weights + 3 biases)")
    for name, rule, shape, torch_shape, diff, _ in results:
        print(f"{name:<22} rule {rule}x{rule}  loops {shape}  torch {torch_shape}"
              f"  max|loops-torch| = {diff}")
    y = results[0][5]
    for o in range(3):
        print(f"stride 1: y[0, {o}] =")
        grid(y[o * 9:], 3, 3)


# Output channels 0, 1, 2 of the stride-1 case, row by row (Burn gives the same).
EXPECTED_Y0 = [
    -1.0, -5.0, 5.0, 1.0, -35.0, 20.0, 10.0, -12.0, 20.0,
    -10.0, 7.0, 4.0, 6.0, 3.0, -10.0, -15.0, -3.0, 19.0,
    13.0, 9.0, -17.0, -11.0, 2.0, -38.0, 1.0, 32.0, -18.0,
]


def test_matches_burn():
    _, _, results, params = build()
    assert params == 57
    assert [r[1] for r in results] == [3, 5, 3, 5]
    assert all(r[2] == r[3] for r in results)
    assert all(r[4] == 0.0 for r in results)
    assert results[0][5] == EXPECTED_Y0


if __name__ == "__main__":
    main()
