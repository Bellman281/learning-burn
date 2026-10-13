"""Example 11.4 - Pooling by Hand (PyTorch reference + parity test).

Max and average pooling written as loops over the windows, recording WHERE
each maximum came from. Checked against F.max_pool2d(return_indices=True)
and F.avg_pool2d.

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

    def below(self, n):
        return self.next() % n


def shuffled(n, rng):
    """0..n-1 in a random order (Fisher-Yates), so no two pixels tie."""
    v = [float(i) for i in range(n)]
    for i in range(n - 1, 0, -1):
        j = rng.below(i + 1)
        v[i], v[j] = v[j], v[i]
    return v


def max_pool(x, h, w, k, s):
    """Returns (maxima, flat argmax index r*w+q of each window, output side)."""
    ho, wo = (h - k) // s + 1, (w - k) // s + 1
    vals, idx = [], []
    for i in range(ho):
        for j in range(wo):
            best, at = float("-inf"), 0
            for a in range(k):
                for e in range(k):
                    r, q = i * s + a, j * s + e
                    if x[r * w + q] > best:
                        best, at = x[r * w + q], r * w + q
            vals.append(best)
            idx.append(at)
    return vals, idx, ho


def avg_pool(x, h, w, k, s):
    ho, wo = (h - k) // s + 1, (w - k) // s + 1
    return [sum(x[(i * s + a) * w + j * s + e] for a in range(k) for e in range(k)) / (k * k)
            for i in range(ho) for j in range(wo)]


def receptive_fields(layers):
    rf, jump, out = 1, 1, []
    for name, k, stride in layers:
        rf += (k - 1) * jump
        jump *= stride
        out.append((name, rf, jump))
    return out


def build():
    x = shuffled(36, Lcg(10))
    xt = torch.tensor(x).reshape(1, 1, 6, 6)
    results = {}
    for name, k, s in [("max 2x2/2", 2, 2), ("max 3x3/2", 3, 2)]:
        vals, idx, _ = max_pool(x, 6, 6, k, s)
        tv, ti = F.max_pool2d(xt, k, s, return_indices=True)
        results[name] = (vals, idx, tv.flatten().tolist() == vals, ti.flatten().tolist() == idx)
    avg = avg_pool(x, 6, 6, 2, 2)
    avg_ok = F.avg_pool2d(xt, 2, 2).flatten().tolist() == avg
    rf = receptive_fields([("conv 3x3", 3, 1), ("pool 2x2", 2, 2), ("conv 3x3", 3, 1)])
    return x, results, avg, avg_ok, rf


def main():
    x, results, avg, avg_ok, rf = build()
    print("x =")
    for r in range(6):
        print("   " + "".join(f"{x[r * 6 + c]:5.0f}" for c in range(6)))
    for name, (vals, idx, vals_ok, idx_ok) in results.items():
        where = [f"({i // 6},{i % 6})" for i in idx]
        print(f"{name}: values {vals}")
        print(f"           argmax {idx} = {' '.join(where)}")
        print(f"           same as torch: values {vals_ok}, indices {idx_ok}")
    print(f"avg 2x2/2: {avg}  same as torch: {avg_ok}")
    for name, size, jump in rf:
        print(f"after {name}: receptive field {size}x{size}, jump {jump}")


# The flat argmax indices and averages Burn reports (rust/src/main.rs).
EXPECTED_IDX_2 = [1, 2, 5, 12, 14, 22, 24, 26, 29]
EXPECTED_IDX_3 = [14, 14, 14, 14]
EXPECTED_AVG = [12.75, 11.0, 20.25, 21.75, 22.75, 18.5, 14.75, 22.25, 13.5]


def test_matches_burn():
    x, results, avg, avg_ok, rf = build()
    vals, idx, vals_ok, idx_ok = results["max 2x2/2"]
    assert vals_ok and idx_ok
    assert idx == EXPECTED_IDX_2
    vals3, idx3, ok3v, ok3i = results["max 3x3/2"]
    assert ok3v and ok3i
    assert idx3 == EXPECTED_IDX_3
    assert avg_ok and avg == EXPECTED_AVG
    assert [r[1] for r in rf] == [3, 4, 8]

if __name__ == "__main__":
    main()
