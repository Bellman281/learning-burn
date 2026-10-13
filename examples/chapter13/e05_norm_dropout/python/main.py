"""Example 13.5 - Normalisation and Dropout Layers (PyTorch reference + parity test).

BatchNorm2d in train and eval mode (batch statistics vs running statistics,
and the momentum update), LayerNorm with a learned scale and shift, and
Dropout in train and eval mode.

Run:   python main.py
Test:  pytest main.py
"""

import torch
import torch.nn as nn


def bn_input():
    return torch.tensor([
        [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]]],
        [[[5.0, 6.0], [7.0, 8.0]], [[3.0, 3.0], [3.0, 3.0]]],
    ])


def batch_norm():
    bn = nn.BatchNorm2d(2)                       # momentum 0.1, eps 1e-5
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([2.0, 0.5]))
        bn.bias.copy_(torch.tensor([1.0, -1.0]))
    bn.train()
    y_train = bn(bn_input()).detach()
    running_mean = bn.running_mean.tolist()
    running_var = bn.running_var.tolist()        # uses the UNBIASED batch variance
    bn.eval()
    y_eval = bn(bn_input()).detach()

    # Burn's convention: the biased variance goes into running_var.
    x = bn_input()
    biased = x.transpose(0, 1).reshape(2, -1).var(dim=1, unbiased=False)
    with torch.no_grad():
        bn.running_var.copy_(0.9 * torch.ones(2) + 0.1 * biased)
    y_eval_burn = bn(bn_input()).detach()
    return (y_train.flatten().tolist(), running_mean, running_var,
            y_eval.flatten().tolist(), bn.running_var.tolist(), y_eval_burn.flatten().tolist())


def layer_norm():
    ln = nn.LayerNorm(4)                          # eps 1e-5
    with torch.no_grad():
        ln.weight.copy_(torch.tensor([1.0, 0.5, 2.0, 1.0]))
        ln.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 1.0]))
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    return ln(x).detach().flatten().tolist()


def dropout():
    torch.manual_seed(42)
    drop = nn.Dropout(0.25)
    drop.train()
    train = drop(torch.ones(1000))
    drop.eval()
    evaluated = drop(torch.ones(1000))
    return train.tolist(), evaluated.tolist()


def fmt(v):
    return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"


def main():
    y_train, mean, var, y_eval, var_burn, y_eval_burn = batch_norm()
    print("BatchNorm2d, training mode (batch statistics):")
    print("  image 0, channel 0 =", fmt(y_train[0:4]))
    print("  image 0, channel 1 =", fmt(y_train[4:8]))
    print("  running_mean =", fmt(mean), "  running_var =", fmt(var))
    print("BatchNorm2d, eval mode (running statistics):")
    print("  image 0, channel 0 =", fmt(y_eval[0:4]))
    print("  image 0, channel 1 =", fmt(y_eval[4:8]))
    print("with Burn's running_var", fmt(var_burn), "the eval output is:")
    print("  image 0, channel 0 =", fmt(y_eval_burn[0:4]))
    print("  image 0, channel 1 =", fmt(y_eval_burn[4:8]))

    ln = layer_norm()
    print("LayerNorm row 0 =", fmt(ln[0:4]))
    print("LayerNorm row 1 =", fmt(ln[4:8]))

    train, evaluated = dropout()
    kept = sum(1 for v in train if v != 0.0)
    print(f"Dropout p=0.25, train(): kept {kept}/1000, kept value {max(train):.4f}, "
          f"mean {sum(train) / 1000:.4f}")
    print("Dropout p=0.25, eval():  output == input:", all(v == 1.0 for v in evaluated))


def close(a, b, tol):
    assert len(a) == len(b)
    assert all(abs(x - y) < tol for x, y in zip(a, b)), (a, b)


def test_matches_burn():
    y_train, mean, var, y_eval, var_burn, y_eval_burn = batch_norm()
    close(y_train[0:8], [-2.0551, -1.1822, -0.3093, 0.5636, -1.5, -1.5, -1.5, -1.5], 1e-4)
    close(mean, [0.45, 0.2], 1e-6)
    close(var, [1.5, 1.0142857], 1e-6)            # PyTorch: unbiased
    close(var_burn, [1.425, 1.0], 1e-6)           # Burn: biased
    close(y_eval_burn[0:4], [1.9215, 3.5969, 5.2723, 6.9477], 1e-4)
    close(y_eval_burn[4:8], [-0.6, -0.6, -0.6, -0.6], 1e-4)
    close(layer_norm(), [-1.3416, -0.2236, 0.8944, 2.3416, -1.3416, -0.2236, 0.8944, 2.3416], 1e-4)
    train, evaluated = dropout()
    assert all(v == 0.0 or abs(v - 4 / 3) < 1e-6 for v in train)
    assert 700 <= sum(1 for v in train if v != 0.0) <= 800
    assert all(v == 1.0 for v in evaluated)


if __name__ == "__main__":
    main()
