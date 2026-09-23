"""Example 20.3 - Micro-batching (Python reference + parity test).

32 client threads x 40 requests against two servers: one forward pass per
request, and a queue + worker that runs requests in batches of up to 32.

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


import queue
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = load_model()
JOBS = queue.Queue()
BATCHES = [0]


def batch_worker(max_batch=32, max_wait=0.002):
    while True:
        jobs = [JOBS.get()]
        deadline = time.perf_counter() + max_wait
        while len(jobs) < max_batch and time.perf_counter() < deadline:
            try:
                jobs.append(JOBS.get_nowait())
            except queue.Empty:
                time.sleep(0)
        x = torch.stack([torch.tensor(p).reshape(1, 12, 12) for p, _ in jobs])
        with torch.no_grad():
            classes = MODEL(x).argmax(1).tolist()
        for (_, box), c in zip(jobs, classes):
            box["class"] = c
            box["done"].set()
        BATCHES[0] += 1


def make_handler(batched):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            pixels = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["pixels"]
            if batched:
                box = {"done": threading.Event()}
                JOBS.put((pixels, box))
                box["done"].wait()
                c = box["class"]
            else:
                with torch.no_grad():
                    c = MODEL(torch.tensor(pixels).reshape(1, 1, 12, 12)).argmax(1).item()
            body = json.dumps({"class": c}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return Handler


class Server(ThreadingHTTPServer):
    request_queue_size = 128   # the default backlog of 5 drops connections under load


def load_test(batched, clients=32, per_client=40):
    server = Server(("127.0.0.1", 0), make_handler(batched))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/predict"

    def client(c):
        rng, out = mw.Lcg(7 + c), []
        for i in range(per_client):
            body = json.dumps({"pixels": mw.make_image(i % 4, rng).tolist()}).encode()
            t = time.perf_counter()
            with urllib.request.urlopen(urllib.request.Request(url, body)) as r:
                cls = json.loads(r.read())["class"]
            out.append(((time.perf_counter() - t) * 1e3, cls == i % 4))
        return out

    start = time.perf_counter()
    with ThreadPoolExecutor(clients) as pool:
        results = [r for rs in pool.map(client, range(clients)) for r in rs]
    seconds = time.perf_counter() - start
    server.shutdown()
    lat = sorted(ms for ms, _ in results)
    return {"rps": len(lat) / seconds, "p50": lat[int((len(lat) - 1) * 0.5)],
            "p99": lat[int((len(lat) - 1) * 0.99)], "correct": sum(ok for _, ok in results),
            "requests": len(lat)}


def build():
    threading.Thread(target=batch_worker, daemon=True).start()
    a = load_test(batched=False)
    BATCHES[0] = 0
    b = load_test(batched=True)
    return a, b, b["requests"] / max(BATCHES[0], 1)


def main():
    a, b, mean_batch = build()
    print("32 clients x 40 requests each")
    print("strategy   requests/s  p50 ms  p99 ms  correct")
    for name, r in (("direct", a), ("batched", b)):
        print(f"{name:<9}  {r['rps']:>10.0f}  {r['p50']:>6.2f}  {r['p99']:>6.2f}  "
              f"{r['correct']}/{r['requests']}")
    print(f"mean batch size in the batched run = {mean_batch:.1f}")


def test_both_strategies_answer_correctly():
    a, b, _ = build()
    assert (a["correct"], b["correct"]) == (1280, 1280)


if __name__ == "__main__":
    main()
