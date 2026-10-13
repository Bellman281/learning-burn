"""Example 13.7 - A Transformer Encoder with burn::nn (PyTorch reference + parity test).

nn.TransformerEncoder (2 post-norm layers, d_model 8, 2 heads, d_ff 16, GELU)
with fixed deterministic weights copied into PyTorch's layout: the three
Burn projections query/key/value become one in_proj_weight. The outputs, with
and without a padding mask, must match Burn's.

Run:   python main.py
Test:  pytest main.py
"""

import math

import torch
import torch.nn as nn

D_MODEL, N_HEADS, D_FF, N_LAYERS, SEQ = 8, 2, 16, 2, 5


class Filler:
    """Every tensor gets its own phase, in the same order as main.rs."""

    def __init__(self):
        self.phase = 0.0

    def wave(self, n, step, scale, offset):
        p = self.phase
        self.phase += 1.0
        return torch.tensor([offset + scale * math.sin(i * step + p) for i in range(n)],
                            dtype=torch.float32)

    def linear(self, d_in, d_out):   # PyTorch layout: [out, in]
        w = self.wave(d_out * d_in, 2.3, 1 / math.sqrt(d_in), 0.0).reshape(d_out, d_in)
        return w, self.wave(d_out, 1.7, 0.1, 0.0)

    def norm(self, ln):
        ln.weight.copy_(self.wave(D_MODEL, 1.7, 0.1, 1.0))
        ln.bias.copy_(self.wave(D_MODEL, 1.7, 0.1, 0.0))


@torch.no_grad()
def encoder():
    layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.1, activation="gelu",
                                       batch_first=True, norm_first=False)
    enc = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)
    f = Filler()
    for layer in enc.layers:
        (wq, bq), (wk, bk), (wv, bv) = (f.linear(D_MODEL, D_MODEL) for _ in range(3))
        layer.self_attn.in_proj_weight.copy_(torch.cat([wq, wk, wv]))   # [3*d, d]
        layer.self_attn.in_proj_bias.copy_(torch.cat([bq, bk, bv]))
        w, b = f.linear(D_MODEL, D_MODEL)
        layer.self_attn.out_proj.weight.copy_(w)
        layer.self_attn.out_proj.bias.copy_(b)
        w, b = f.linear(D_MODEL, D_FF)
        layer.linear1.weight.copy_(w)
        layer.linear1.bias.copy_(b)
        w, b = f.linear(D_FF, D_MODEL)
        layer.linear2.weight.copy_(w)
        layer.linear2.bias.copy_(b)
        f.norm(layer.norm1)
        f.norm(layer.norm2)
    return enc.eval()                # dropout off


def build():
    enc = encoder()
    x = torch.tensor([math.sin(i * 0.37) for i in range(2 * SEQ * D_MODEL)],
                     dtype=torch.float32).reshape(2, SEQ, D_MODEL)
    pad = torch.tensor([[False] * SEQ, [False, False, False, True, True]])
    with torch.no_grad():
        out = enc(x)
        masked = enc(x, src_key_padding_mask=pad)
    params = sum(p.numel() for p in enc.parameters())
    return out.flatten().tolist(), masked.flatten().tolist(), params


EXPECTED = [
    -1.900696, -0.776611, -0.089673, 0.574809, 1.192954, 1.009292, 0.309706, -0.235800,
    1.902933, 1.080650, -0.038460, -0.649469, -0.752175, -0.916261, -0.796297, -0.083141,
    -1.835049, -1.046998, -0.414450, 0.193011, 0.799110, 0.959245, 0.791732, 0.594326,
    1.763327, 1.210660, 0.359623, -0.261842, -0.560472, -0.911393, -1.041491, -0.745952,
    -1.651943, -1.100640, -0.569386, -0.040457, 0.534989, 0.861169, 0.976852, 0.996408,
    1.509552, 1.223787, 0.614345, 0.050147, -0.336066, -0.819432, -1.170676, -1.203137,
    -1.532384, -1.104985, -0.637037, -0.148770, 0.403432, 0.799949, 1.041690, 1.166564,
    1.296498, 1.193802, 0.742690, 0.235523, -0.172030, -0.727866, -1.218533, -1.445673,
    -1.344964, -1.076711, -0.736314, -0.306242, 0.249662, 0.714350, 1.086424, 1.372861,
    1.055725, 1.140356, 0.830514, 0.401221, 0.012618, -0.610841, -1.248665, -1.643839,
]
EXPECTED_MASKED = [
    -1.900696, -0.776611, -0.089673, 0.574809, 1.192954, 1.009292, 0.309707, -0.235800,
    1.902933, 1.080650, -0.038460, -0.649469, -0.752175, -0.916261, -0.796297, -0.083141,
    -1.835049, -1.046998, -0.414450, 0.193011, 0.799110, 0.959245, 0.791732, 0.594326,
    1.763327, 1.210660, 0.359623, -0.261842, -0.560472, -0.911393, -1.041491, -0.745952,
    -1.651943, -1.100640, -0.569386, -0.040457, 0.534989, 0.861169, 0.976852, 0.996408,
    1.483188, 1.221516, 0.630854, 0.076267, -0.312558, -0.807389, -1.180475, -1.238461,
    -1.553754, -1.107697, -0.623729, -0.127084, 0.422228, 0.809563, 1.034302, 1.138179,
    1.264892, 1.186993, 0.756765, 0.260942, -0.147899, -0.713009, -1.223629, -1.476012,
    -1.370672, -1.082477, -0.724793, -0.284766, 0.269875, 0.726278, 1.081759, 1.347557,
    1.019168, 1.129073, 0.842076, 0.425803, 0.037216, -0.593339, -1.249122, -1.668994,
]


def fmt(v):
    return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"


def main():
    out, masked, params = build()
    print("seq 0, token 0 =", fmt(out[0:8]))
    print("seq 1, token 4 =", fmt(out[72:80]))
    print("with padding mask, seq 1, token 0 =", fmt(masked[40:48]))
    print("parameters =", params)


def test_matches_burn():
    out, masked, params = build()
    assert all(abs(a - b) < 1e-5 for a, b in zip(out, EXPECTED))
    assert all(abs(a - b) < 1e-5 for a, b in zip(masked, EXPECTED_MASKED))
    assert params == 1200


if __name__ == "__main__":
    main()
