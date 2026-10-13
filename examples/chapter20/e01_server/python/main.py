"""Example 20.1 - An Inference Server (Python reference + parity test).

The same service with the standard library's http.server: load the model
once, POST /predict with {"pixels": [...]}, get {"class", "logits"} back.
(In production you would reach for FastAPI; the shape is identical.)

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


import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = load_model()  # loaded once, shared by every request


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body, ctype="application/json"):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, "ok", "text/plain") if self.path == "/health" else self.reply(404, "")

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        pixels = req["pixels"]
        if len(pixels) != 144:
            return self.reply(422, f"expected 144 pixels, got {len(pixels)}", "text/plain")
        with torch.no_grad():
            logits = MODEL(torch.tensor(pixels).reshape(1, 1, 12, 12))[0].tolist()
        self.reply(200, json.dumps({"class": max(range(4), key=lambda i: logits[i]), "logits": logits}))


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def build():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    health = urllib.request.urlopen(url + "/health").read().decode()
    rng = mw.Lcg(7)
    classes = [post(url + "/predict", {"pixels": mw.make_image(i % 4, rng).tolist()})[1]["class"]
               for i in range(8)]
    bad, _ = post(url + "/predict", {"pixels": [0.0] * 10})
    server.shutdown()
    return health, classes, bad


def main():
    health, classes, bad = build()
    print("GET /health ->", health)
    print("classes for the 8 test images =", classes)
    print("a request with 10 pixels -> HTTP", bad)


def test_matches_burn():
    assert build() == ("ok", [0, 1, 2, 3, 0, 1, 2, 3], 422)


if __name__ == "__main__":
    main()
