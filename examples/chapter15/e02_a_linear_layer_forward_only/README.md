# A Linear Layer, Served (Example 15.2)

The smallest complete picture of what **inference** actually is: a matmul, a bias, an activation. No gradients, no optimizer, no training loop.

## The Core Concept
Every chapter before this one was concerned with *producing* weights. This example consumes them. The weights arrived from somewhere else — a training run, a checkpoint, a HuggingFace download — and the only job left is to push one input through them:

$$y = \text{ReLU}(x\mathbf{W} + b)$$

That is the entire forward pass of a served model, and you can check it by hand. With $x = [1, 2, 3]$:

- $1(0.1) + 2(0.3) + 3(0.5) = 2.2$
- $1(0.2) + 2(0.4) + 3(0.6) = 2.8$

Add the bias $[0.5, -0.5]$ and you have $[2.7, 2.3]$. Both positive, so `relu` leaves them alone.

## Broadcasting Is Explicit
Note `broadcast_add`. Candle will **not** silently stretch the `[2]` bias across a `[1, 2]` result; you have to say that is what you meant. Burn made you write `unsqueeze()` for exactly the same reason back in Chapter 2. Two different frameworks, two different mechanisms, one shared refusal to guess your intent.

## Key Concepts Covered
- **Inference vs. Training:** Recognizing a forward pass as the whole of what a deployed model does.
- **Explicit Broadcasting:** Using `broadcast_add` to align a rank-1 bias with a rank-2 activation.
- **Method Chaining Through `Result`:** `x.matmul(&w)?.broadcast_add(&b)?.relu()?` — each link fallible, each handled.
- **Hand-Checkable Arithmetic:** Weights chosen small enough that you can verify the output on paper.

## Running It
```bash
cargo run -- -c 15 -e 2 rust
cargo run -- -c 15 -e 2 python
```

Rust prints `[[2.7, 2.3000002]]`. The $2.3000002$ is not a bug — it is $f32$ telling the truth about what the machine did.
