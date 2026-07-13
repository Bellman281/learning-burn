# Learning Burn

[![Burn](https://img.shields.io/badge/burn-0.21.0-3b5b92)](https://github.com/tracel-ai/burn)
[![Rust](https://img.shields.io/badge/rust-2024_edition-dea584?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-omidpolyO-FFDD00?logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/omidpolyO)

Deep learning in Rust, for people who already do it in Python.

Runnable code for every idea in the *Learning Burn* book — tensors, autodiff,
backprop, transformers, quantisation, bare-metal inference — each a single file
you can `cargo run` and read.

---

## Why would a data scientist touch this?

Most of you shouldn't, most of the time. If your job is EDA, feature work, and
iterating on models in a notebook, **stay in Python.** The tooling is better, the
ecosystem is thirty times larger, and nothing here will make you faster at that.

Rust becomes the right answer at exactly one moment: **when the model has to leave
your laptop.**

| Your problem | What Rust gives you |
|---|---|
| "Inference is fast, but the *service* is slow" | No interpreter, no GIL, no GC pauses. Predictable p99 latency. |
| "Deployment is a 4 GB Docker image and a dependency hell" | One static binary. No Python, no CUDA runtime, no `requirements.txt`. |
| "It needs to run on the device, not the cloud" | `no_std`. Weights compiled into the firmware. **A 2,410-parameter model is 2.4 KB quantised** — it fits on a microcontroller with 256 KB of SRAM. |
| "It fails at 3 a.m. with a shape error" | Tensor rank is part of the type. `Tensor<B, 2>` and `Tensor<B, 3>` are different types; mixing them doesn't compile. |
| "Two teams: researchers in Python, engineers rewriting in C++" | One language for both. The training code and the deployed code are the same code. |

If none of those are your problem, close the tab — with our blessing. If one of
them is the reason your last project stalled, this repo is worth an afternoon.

**What Rust will not give you:** a mature ecosystem, pretrained model zoos,
`pandas`, or a fast iteration loop. Burn is good. It is not PyTorch, and pretending
otherwise would waste your time.

---

## Sixty seconds

```bash
git clone https://github.com/jhosein58/learning-burn
cd learning-burn/book-tests

cargo run --example ch5_01_backward_gradient    # autodiff: df/dx of sum(x²) is 2x
cargo run --example ch9_01_sgd_training         # a real training loop, 20 lines
cargo run --example ch10_02_two_layer_backprop  # gradients through a hidden layer
cargo run --example ch10_03_xor_mlp             # watch it learn XOR from scratch
```

No GPU needed — everything runs on the CPU (`ndarray`) backend.

Attention, transformers, quantisation and bare-metal inference are in open PRs and
land shortly; see **Status** below.

---

## The three things that will confuse you first

Coming from PyTorch, these are the ones that actually cost people a day:

**1. `backward()` returns the gradients. It does not attach them.**

```rust
let grads = loss.backward();      // <- a container, RETURNED
let dw = w.grad(&grads).unwrap(); // <- you look it up
```

No `.grad` on the tensor, and therefore **no `zero_grad()`** — there is nothing to
accumulate into, so the classic "forgot to zero the gradients" bug is unwriteable.

**2. There is no `no_grad()` block.**

Autodiff is a *wrapper around the backend*, not a mode you enter.
`Autodiff<NdArray>` tracks gradients; `NdArray` doesn't. To do inference, you drop
the wrapper — `x.inner()` — or just type your model over the plain backend. There is
no `model.eval()` to forget.

**3. Operations consume their inputs.**

```rust
let c = a.clone() + b;   // `a` is moved unless you clone it
```

Every `.clone()` you see is the code saying *"I need this value again."* It looks
like noise for about a chapter, then it starts looking like documentation.

---

## What's in the sixteen chapters

**Foundations** — Rust for Python people (ownership, borrowing, traits — no Burn
yet) · tensors · elementwise ops and broadcasting · matmul and the shape rule ·
norms, determinant, Gram matrix · autodiff.

**Learning** — linear regression by hand · activations and *why ReLU beat sigmoid*
(it's one number: 0.25) · MSE and cross-entropy · SGD and the learning rate ·
**backpropagation** — the chain rule with numbers small enough to check on paper ·
a network built from scratch, with the gradients derived by hand and then verified
against autodiff.

**Building** — attention, causal masking, multi-head, a full transformer block.

**Shipping** — save/load and the f32-vs-f16 trade-off · bare-metal: weights baked
into the binary, backend-agnostic inference, int8 quantisation, memory footprint ·
running pretrained models with Candle, and when to use it instead of Burn.

Full list in [`book-tests/examples/`](book-tests/examples/).

---

## Status

**Code:** 43 of 60 examples merged; the rest are open PRs.

**Book:** first full draft complete — 16 chapters, 185 pages, 61 figures. Not
edited, not published. **Watch this repo** and you'll know when it is.

The book has no copy of the code: every listing is read out of *this repository* at
build time, and every program output it quotes comes from a file produced by
actually running that example. So the book cannot drift from the code, and the
examples you run here are the ones you'll read there.

---

## Contributing

**An example that doesn't run is a bug — file it, and paste the output.** That is
the report we most want.

If you want something meatier: `burn::tensor::linalg` ships `lu` (LU decomposition
with partial pivoting) but has **no `solve`**. LU is exactly what a `solve` is built
from — factor `A = PLU`, then forward- and back-substitute. The hard, numerically
delicate part is already in the library; the substitution pass on top is not. It's a
well-scoped contribution to Burn itself, and it's wanted:
[burn#1538](https://github.com/tracel-ai/burn/issues/1538).

---

## Credit

Built on [**Burn**](https://github.com/tracel-ai/burn) (pinned to 0.21.0). The
framework is theirs; this repo is original example code against its public API.
**If this is useful to you, star [the Burn team](https://github.com/tracel-ai/burn),
not us.** They did the hard part.

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Copyright © 2026 **H0531N** and
**Bellman281**. If you adapt these examples, please attribute
<https://github.com/jhosein58/learning-burn> — see [`NOTICE`](NOTICE).
