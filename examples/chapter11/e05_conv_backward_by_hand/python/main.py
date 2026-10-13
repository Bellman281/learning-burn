"""Example 11.5 - The Convolution Backward Pass by Hand (PyTorch reference + parity test).

conv (2 -> 3 channels, 3x3) + bias -> 2x2 max-pool -> L = sum(pooled * R).
The three gradients written out by hand -- bias, weights, input -- plus the
max-pool routing, checked against torch .backward(). The data are multiples
of 1/4, so every sum is exact and the two agree bit for bit.

Run:   python main.py
Test:  pytest main.py
"""

import torch
import torch.nn.functional as F

N, CI, CO, H, K = 2, 2, 3, 6, 3
HO = H - K + 1    # 4
HP = HO // 2      # 2


class Lcg:
    """Tiny LCG, bit-identical to the Rust one."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF

    def quarter(self):
        return (float(self.next() % 9) - 4.0) / 4.0   # a multiple of 1/4 in [-1, 1]


def data():
    rng = Lcg(2)
    x = torch.tensor([rng.quarter() for _ in range(N * CI * H * H)]).reshape(N, CI, H, H)
    w = torch.tensor([rng.quarter() for _ in range(CO * CI * K * K)]).reshape(CO, CI, K, K)
    b = torch.tensor([0.5, -0.25, 0.0])
    r = torch.tensor([rng.quarter() for _ in range(N * CO * HP * HP)]).reshape(N, CO, HP, HP)
    return x, w, b, r


def corr2d(img, ker):
    """'Valid' cross-correlation of one 2-D map with one 2-D kernel."""
    kh, kw = ker.shape
    oh, ow = img.shape[0] - kh + 1, img.shape[1] - kw + 1
    out = torch.zeros(oh, ow)
    for a in range(kh):
        for e in range(kw):
            out += ker[a, e] * img[a:a + oh, e:e + ow]
    return out


def by_hand(x, w, b, r):
    # forward
    y = torch.zeros(N, CO, HO, HO)
    for n in range(N):
        for o in range(CO):
            y[n, o] = b[o] + sum(corr2d(x[n, c], w[o, c]) for c in range(CI))
    gy = torch.zeros_like(y)
    loss = 0.0
    for n in range(N):
        for o in range(CO):
            for i in range(HP):
                for j in range(HP):
                    win = y[n, o, 2 * i:2 * i + 2, 2 * j:2 * j + 2].flatten()
                    k = int(torch.argmax(win))
                    a, e = divmod(k, 2)
                    loss += float(win[k] * r[n, o, i, j])
                    gy[n, o, 2 * i + a, 2 * j + e] += r[n, o, i, j]   # max-pool routing
    db = gy.sum(dim=(0, 2, 3))                                       # bias
    dw = torch.zeros(CO, CI, K, K)
    for o in range(CO):
        for c in range(CI):
            dw[o, c] = sum(corr2d(x[n, c], gy[n, o]) for n in range(N))  # weights
    dx = torch.zeros(N, CI, H, H)
    for n in range(N):
        for c in range(CI):
            dx[n, c] = sum(corr2d(F.pad(gy[n, o], (K - 1,) * 4), w[o, c].flip(0, 1))
                           for o in range(CO))                       # input
    return loss, gy, db, dw, dx


def by_torch(x, w, b, r):
    x, w, b = (t.clone().requires_grad_() for t in (x, w, b))
    pooled = F.max_pool2d(F.conv2d(x, w, b), 2)
    loss = (pooled * r).sum()
    loss.backward()
    return loss.item(), b.grad, w.grad, x.grad


def build():
    x, w, b, r = data()
    hand = by_hand(x, w, b, r)
    auto = by_torch(x, w, b, r)
    return x, w, r, hand, auto


def show(name, t):
    print(f"{name} =")
    for row in t.reshape(-1, t.shape[-1]).tolist():
        print("  " + "".join(f"{v:8.4f}" for v in row))


def main():
    x, w, r, (loss, gy, db, dw, dx), (t_loss, t_db, t_dw, t_dx) = build()
    print(f"x {[N, CI, H, H]}  W {[CO, CI, K, K]}  y {[N, CO, HO, HO]}  pooled {[N, CO, HP, HP]}")
    print(f"loss: by hand {loss}   torch {t_loss}")
    show("x[0, 0]", x[0, 0])
    show("W[0, 0]", w[0, 0])
    show("R[0, 0]   (dL/dpooled)", r[0, 0])
    show("gy[0, 0]  (routed to the argmax)", gy[0, 0])
    print("db  by hand", db.tolist())
    print("db  torch  ", t_db.tolist())
    show("dW[0, 0] by hand", dw[0, 0])
    show("dx[0, 0] by hand", dx[0, 0])
    print("max|hand - torch|: db", (db - t_db).abs().max().item(),
          " dW", (dw - t_dw).abs().max().item(), " dx", (dx - t_dx).abs().max().item())
    print("sum(dx) =", dx.sum().item())


def test_matches_burn():
    _, _, _, (loss, gy, db, dw, dx), (t_loss, t_db, t_dw, t_dx) = build()
    assert loss == t_loss
    assert torch.equal(db, t_db) and torch.equal(dw, t_dw) and torch.equal(dx, t_dx)
    # The same numbers Burn's autodiff gives in rust/src/main.rs.
    assert loss == 0.375 and db.tolist() == [1.25, -0.25, -0.75]
    assert dw[0, 0].flatten().tolist() == [-1.25, 0.875, -0.75, 0.5, -0.9375, -0.9375,
                                           -0.5625, 1.5625, 0.5625]
    assert dx.sum().item() == -0.1875


if __name__ == "__main__":
    main()
