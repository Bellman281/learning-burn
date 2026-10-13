"""Example 20.2 - Batch Size and Throughput (PyTorch reference + parity test).

The same forward pass at batch sizes 1..256. Timings are machine-dependent;
the test checks that batching changes the speed and not the answers.

Run:   python main.py
Test:  pytest main.py
"""

import importlib.util
import json
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "..", "..", "chapter19", "weights")

# Chapter 19's PyTorch model and data generator.
_spec = importlib.util.spec_from_file_location("make_weights", os.path.join(WEIGHTS, "make_weights.py"))
mw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mw)


def load_model():
    from safetensors.torch import load_file
    model = mw.TorchCnn()
    model.load_state_dict(load_file(os.path.join(WEIGHTS, "small_cnn.safetensors")))
    return model.eval()


def images(n):
    rng = mw.Lcg(7)
    return torch.stack([torch.from_numpy(mw.make_image(i % 4, rng)).reshape(1, 12, 12)
                        for i in range(n)])


@torch.no_grad()
def time_batch(model, n):
    x = images(n)
    reps = max(8, min(400, 4096 // n))
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t) * 1e6)
    return sorted(times)[len(times) // 2]


@torch.no_grad()
def build():
    model = load_model()
    rows = [(n, time_batch(model, n)) for n in (1, 4, 16, 64, 256)]
    x = images(64)
    all_at_once = model(x)
    one_by_one = torch.cat([model(x[i:i + 1]) for i in range(64)])
    return rows, (all_at_once - one_by_one).abs().max().item()


def main():
    rows, diff = build()
    print("batch  us per batch  us per image  images/s")
    for n, us in rows:
        print(f"{n:>5}  {us:>12.1f}  {us / n:>12.2f}  {n / us * 1e6:>8.0f}")
    print(f"max |batched - one by one| = {diff:e}")


def test_batching_changes_speed_not_answers():
    rows, diff = build()
    assert len(rows) == 5
    assert diff < 1e-5


if __name__ == "__main__":
    main()
