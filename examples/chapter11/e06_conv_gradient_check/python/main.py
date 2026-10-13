"""Example 11.6 - Gradient-Checking a Convolution (PyTorch reference + parity test).

conv (2 -> 2 channels, 3x3, padding 1) + bias, loss L = sum(R * tanh(y)).
The hand-written backward pass from Example 11.5 (now with padding) against
central finite differences, everything in float64. A deliberately broken
backward -- the kernel NOT rotated -- fails the same check.

Run:   python main.py
Test:  pytest main.py
"""

import math

import torch
import torch.nn.functional as F

DT = torch.float64   # a gradient check runs in f64 (Chapter 9)
N, CI, CO, H, K, P = 1, 2, 2, 5, 3, 1
EPS = 1e-5


class Lcg:
    """Tiny LCG, bit-identical to the Rust one."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF

    def uniform(self):
        return self.next() / 32768.0 * 2.0 - 1.0   # in [-1, 1), exact in f64


def data():
    rng = Lcg(9)
    draw = lambda *shape: torch.tensor([rng.uniform() for _ in range(math.prod(shape))],
                                       dtype=DT).reshape(shape)
    x, w, b, r = draw(N, CI, H, H), draw(CO, CI, K, K), draw(CO), draw(N, CO, H, H)
    return x, w, b, r


def corr2d(img, ker):
    kh, kw = ker.shape
    oh, ow = img.shape[0] - kh + 1, img.shape[1] - kw + 1
    out = torch.zeros(oh, ow, dtype=DT)
    for a in range(kh):
        for e in range(kw):
            out += ker[a, e] * img[a:a + oh, e:e + ow]
    return out


def loss_at(x, w, b, r):
    return (r * torch.tanh(F.conv2d(x, w, b, padding=P))).sum().item()


def by_hand(x, w, b, r, rotate=True):
    y = F.conv2d(x, w, b, padding=P)
    gy = r * (1 - torch.tanh(y) ** 2)                 # dL/dy, through tanh
    db = gy.sum(dim=(0, 2, 3))
    dw = torch.zeros_like(w)
    for o in range(CO):
        for c in range(CI):
            dw[o, c] = sum(corr2d(F.pad(x[n, c], (P,) * 4), gy[n, o]) for n in range(N))
    dx = torch.zeros_like(x)
    for n in range(N):
        for c in range(CI):
            ker = lambda o: w[o, c].flip(0, 1) if rotate else w[o, c]
            # full correlation, cropped by the forward padding: pad gy by K-1-P
            dx[n, c] = sum(corr2d(F.pad(gy[n, o], (K - 1 - P,) * 4), ker(o)) for o in range(CO))
    return db, dw, dx


def numerical(x, w, b, r):
    grads = []
    for which in range(3):
        params = [x, w, b]
        g = torch.zeros_like(params[which]).flatten()
        for i in range(g.numel()):
            plus = [p.clone() for p in params]
            minus = [p.clone() for p in params]
            plus[which].view(-1)[i] += EPS
            minus[which].view(-1)[i] -= EPS
            g[i] = (loss_at(*plus, r) - loss_at(*minus, r)) / (2 * EPS)
        grads.append(g.reshape(params[which].shape))
    dx, dw, db = grads
    return db, dw, dx


def rel_err(a, b):
    return ((a - b).abs() / torch.maximum(torch.maximum(a.abs(), b.abs()),
                                          torch.tensor(1e-12, dtype=DT))).max().item()


def build():
    x, w, b, r = data()
    hand = by_hand(x, w, b, r)
    num = numerical(x, w, b, r)
    buggy = by_hand(x, w, b, r, rotate=False)
    xa, wa, ba = (t.clone().requires_grad_() for t in (x, w, b))
    (r * torch.tanh(F.conv2d(xa, wa, ba, padding=P))).sum().backward()
    auto = (ba.grad, wa.grad, xa.grad)
    errs = [rel_err(h, n) for h, n in zip(hand, num)]
    auto_errs = [rel_err(h, a) for h, a in zip(hand, auto)]
    bug_err = rel_err(buggy[2], num[2])
    gradcheck = torch.autograd.gradcheck(
        lambda x, w, b: (r * torch.tanh(F.conv2d(x, w, b, padding=P))).sum(), (xa, wa, ba))
    return loss_at(x, w, b, r), hand, num, auto, errs, auto_errs, bug_err, gradcheck


def main():
    loss, hand, num, auto, errs, auto_errs, bug_err, gradcheck = build()
    print(f"loss = {loss:.12f}")
    print(f"checked {sum(t.numel() for t in hand)} gradient entries, eps = {EPS}")
    for name, i in [("dL/db[0]       ", 0), ("dL/dW[0,0,0,0] ", 1), ("dL/dx[0,0,0,0] ", 2)]:
        h, a, n = (t[i].flatten()[0].item() for t in (hand, auto, num))
        print(f"{name} hand {h:.12f}  torch {a:.12f}  numerical {n:.12f}")
    for name, e, a in zip(["db", "dW", "dx"], errs, auto_errs):
        print(f"max relative error {name}: vs finite differences {e:.1e}, vs autograd {a:.1e}")
    print(f"kernel NOT rotated: max relative error dx = {bug_err:.2f}")
    print(f"torch.autograd.gradcheck: {gradcheck}")


def test_matches_burn():
    loss, hand, num, auto, errs, auto_errs, bug_err, gradcheck = build()
    assert max(errs) < 1e-7 and max(auto_errs) < 1e-12
    assert bug_err > 0.1
    assert gradcheck
    # The same loss and first entries as Burn's f64 autodiff in rust/src/main.rs.
    assert abs(loss - -2.542353235219) < 1e-11
    firsts = [hand[0][0].item(), hand[1].flatten()[0].item(), hand[2].flatten()[0].item()]
    for a, b in zip(firsts, [-5.416377452111, 0.262925182681, -0.150826076058]):
        assert abs(a - b) < 1e-11


if __name__ == "__main__":
    main()
