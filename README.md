# Learning Burn — Code Examples

[![Burn](https://img.shields.io/badge/burn-0.21.0-3b5b92)](https://github.com/tracel-ai/burn)
[![Rust](https://img.shields.io/badge/rust-2024_edition-dea584?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-omidpolyO-FFDD00?logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/omidpolyO)

Compilable, **executed** code examples that accompany the *Learning Burn* book
(by H0531N and Bellman281).

Sixty small, self-contained programs that take you from `let x = 5;` to a
transformer block and an int8 model running bare-metal — one runnable file at a
time, each with a PyTorch equivalent in the comments.

---

## Every example is *run*, not just compiled

This is the rule the repository is built on, and it is not pedantry.

Burn puts tensor rank in the type system, which catches a great deal at compile
time — but not everything. Sizes are values, not types, and const generics are
not arithmetic-checked. So this compiles cleanly:

```rust
let m = Tensor::<B, 2>::from_floats([[4.0, 3.0], [6.0, 3.0]], &dev);
let d = linalg::det::<B, 2, 2, 1>(m);   // ← compiles. panics at runtime.
```

`linalg::det` requires rank ≥ 3 (a *batch* of matrices) and enforces
`D1 == D-1`, `D2 == D-2`. The call above violates all three — and
`cargo build --examples` stays green the whole way. We shipped that bug, and we
only caught it because we run every example and read the output.

**If an example is in this repo, someone has run it and looked at what it
printed.** A teaching repository that panics on first execution is worse than no
repository at all.

The same rule governs the book. Every code listing in it is read out of *this
repository* at build time — the book keeps no copy of the code, so it cannot
drift from it. And every program output the book quotes is read from a file
produced by actually running that example. Neither is ever typed by hand. We
learned that one the hard way too: a draft of Chapter 10 quoted a gradient of
`-0.028` where the program prints `-0.12`, because a plausible-looking number
got typed instead of run. The build now refuses to produce a chapter whose
outputs were never captured.

---

## Who this is for

You know Python and some PyTorch or NumPy. You do not need to know Rust —
**Chapter 0 is a Rust primer written specifically for Python people**, and it
assumes nothing.

You do need to be willing to argue with the borrow checker for about a week.
Chapter 0 will make that week considerably shorter.

---

## Quick start

```bash
git clone https://github.com/jhosein58/learning-burn
cd learning-burn/book-tests

cargo run --example ch1_02_basic_creation   # your first tensor
cargo run --example ch5_01_backward_gradient # your first gradient
cargo build --examples                       # compile all of them
```

No GPU required. Everything runs on the `ndarray` CPU backend by default.

---

## PyTorch → Burn, at a glance

The single most useful page for anyone arriving from Python:

| PyTorch | Burn | Note |
|---|---|---|
| `torch.tensor([[1., 2.]])` | `Tensor::<B, 2>::from_floats([[1., 2.]], &dev)` | rank `2` is part of the **type** |
| `a @ b` | `a.matmul(b)` | `*` is element-wise in both |
| `a * 2.0` | `a.mul_scalar(2.0)` | scalars get their own method |
| `m.T` | `m.transpose()` | or `.t()` |
| `m[0:2, 1:3]` | `m.slice([0..2, 1..3])` | Python colons → Rust ranges |
| `m + bias` | `m + bias.unsqueeze()` | Burn will not promote rank for you |
| `x.requires_grad_()` | `x.require_grad()` | |
| `loss.backward()`; then `x.grad` | `let grads = loss.backward();`<br>then `x.grad(&grads)` | **Burn returns the gradients** — it does not attach them to the tensor |
| `with torch.no_grad():` | `x.inner()` | no `no_grad` block; peel the `Autodiff` wrapper off |
| `torch.linalg.norm(x)` | `linalg::l2_norm(x, 0)` | needs Burn ≥ 0.18 |

Two of these bite everyone. `backward()` **returns** a container you look up,
rather than hanging `.grad` on the tensor. And there is no `no_grad` block,
because `Autodiff<NdArray>` is a *wrapper* you take off with `.inner()`, not a
mode you enter.

---

## Progress

**Book: 11 of 16 chapters drafted.  Examples: 43 of 60 merged.**

The two move independently — a chapter is only drafted once its examples exist
and have been run, so the code lands first and the prose follows.

| | Chapter | Book | Examples | |
|---|---|---|---|---|
| **0** | Rust for Python People | ✍️ drafted | 9 | ✅ merged |
| **1** | Creating Tensors | ✍️ drafted | 9 | ✅ merged |
| **2** | Tensor Operations | ✍️ drafted | 6 | ✅ merged |
| **3** | Matrix Algebra | ✍️ drafted | 3 | ✅ merged |
| **4** | Linear Algebra | ✍️ drafted | 2 | ✅ merged |
| **5** | Autodiff | ✍️ drafted | 3 | ✅ merged |
| **6** | Linear Regression | ✍️ drafted | 2 | ✅ merged |
| **7** | Activation Functions | ✍️ drafted | 3 | ✅ merged |
| **8** | Loss Functions | ✍️ drafted | 2 | ✅ merged |
| **9** | Optimizers | ✍️ drafted | 1 | ✅ merged |
| **10** | Backpropagation | ✍️ drafted | 3 | ✅ merged |
| **11** | Neural Network from Scratch | ⬜ planned | 4 | 🔜 in review |
| **12** | Attention & Transformers | ⬜ planned | 4 | 🔜 in review |
| **13** | Deploying a Model | ⬜ planned | 2 | 🔜 in review |
| **14** | Bare-metal / Embedded | ⬜ planned | 4 | 🔜 in review |
| **15** | Inference with Candle | ⬜ planned | 3 | 🔜 in review |

Drafted chapters are written, illustrated, and code-verified — every listing in
them is read straight out of *this repository* at build time, and every output
they quote was captured from an actual `cargo run`. The book cannot drift from
the code, because it does not keep its own copy of it.

The book itself is not published yet. **Watch this repo** and you'll know when
it is.

---

## What's in each chapter

**0 — Rust for Python People.** Variables and `mut`, ownership and moves,
borrowing, iterators (the comprehension, translated), structs and `impl`, enums
and `Option`, `Result` and `?`, traits, generics. No Burn yet — just the
language, and the one idea Python does not have.

**1 — Creating Tensors.** Rank vs shape, creation, int tensors, filled and
structured, random, from custom types, the `TensorData` bridge, ownership and
cloning, display and `check_closeness`.

**2 — Tensor Operations.** Arithmetic, broadcasting, reshaping and slicing,
reductions, feature standardisation, element-wise masking.

**3 — Matrix Algebra.** `matmul` and the shape rule; transpose, `matvec`, `dot`,
`outer`; `eye`/`tril`/`triu`/`trace`; batched matmul.

**4 — Linear Algebra.** L1/L2/L∞ norms, safe normalisation, the determinant, the
Gram matrix and the normal equations.

**5 — Autodiff.** Your first gradient, the gradient *container* (the big
difference from PyTorch), and `.inner()` for inference.

**6 — Linear Regression.** Gradient descent derived by hand, then checked
against autodiff. They agree exactly — that is the point.

**7 — Activation Functions.** The ReLU family, the saturating ones, softmax.

**8 — Loss Functions.** MSE, cross-entropy.

**9 — Optimizers.** A real SGD training loop.

**10 — Backpropagation.** The chain rule, two-layer backprop by hand, an XOR
MLP. *The chapter that exists because a million people are stuck on this.*

**11 — Neural Network from Scratch.** Forward pass, manual vs autodiff backward,
training from scratch, and a numerical gradient check.

**12 — Attention & Transformers.** Scaled dot-product attention, causal masking,
multi-head attention, a full transformer block.

**13 — Deploying a Model.** Save/load with recorders; the f32 vs f16 precision
trade-off, measured honestly.

**14 — Bare-metal / Embedded.** Weights embedded in the binary, backend-agnostic
inference, int8 quantisation, memory footprint.

**15 — Inference with Candle.** Tensor basics, a linear forward pass, MLP +
softmax — in the separate `candle-inference/` crate.

---

## Contributing

Issues and PRs are welcome, especially:

- **An example that doesn't run.** This is the most valuable bug report you can
  file. Paste the output.
- **A clearer example.** If a listing took you three reads, it is our fault.
- **A missing PyTorch equivalent.** Every example should have one in the
  comments.

If you're looking for something meatier, `burn::tensor::linalg` ships `lu`
(LU decomposition with partial pivoting) but has **no `solve`** — and LU is
exactly what a `solve` is built from. Factor `A = PLU`, then forward- and
back-substitute. The hard part is already in the library; the substitution pass
is not. That is a well-scoped, genuinely wanted contribution to Burn itself
(see [burn#1538](https://github.com/tracel-ai/burn/issues/1538)).

Please run `cargo fmt` and make sure your example **runs**, not just builds.

---

## Built on Burn

These examples are written for [**Burn**](https://github.com/tracel-ai/burn), the
Rust deep learning framework, pinned to **0.21.0**. All credit for the framework
goes to the Burn authors and contributors. This repository contains **original
example code** demonstrating Burn's public API — it does not copy Burn's source.

If this repo is useful to you, the people who deserve your star are
[the Burn team](https://github.com/tracel-ai/burn). Go give them one.

---

## License & attribution

Licensed under the Apache License, Version 2.0 — see [`LICENSE`](LICENSE).

Copyright © 2026 **H0531N** and **Bellman281**.

These examples accompany the *Learning Burn* book. If you use or adapt them,
please attribute this repository:
<https://github.com/jhosein58/learning-burn> — see [`NOTICE`](NOTICE).
