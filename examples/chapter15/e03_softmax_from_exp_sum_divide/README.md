# A Hand-Rolled Softmax (Example 15.3)

A full classifier inference path in Candle — hidden layer, logits, probabilities — where the softmax has to be built from `exp`, `sum` and `divide` because `candle-core` does not ship one. It produces the last number in the book, and that number is not $1.0$.

## The Core Concept
The complete path a served classifier runs:

1. **Hidden Layer:** $h = \text{ReLU}(x\mathbf{W_1} + b_1)$ — $[1, 3] \to [1, 2]$
2. **Logits:** $\text{logits} = h\mathbf{W_2}$ — raw, unnormalized class scores, $[1, 3]$
3. **Softmax:** $\text{probs} = \exp(\text{logits}) / \sum \exp(\text{logits})$ — normalized, $[1, 3]$

There is no `softmax` in `candle-core`, so step 3 is assembled by hand: exponentiate, sum along the row with `sum_keepdim`, divide with `broadcast_div`. That is precisely the Chapter 7 definition, and by now you can write it without looking it up.

## What `exp` Does to a Lead
One class takes the overwhelming share of the mass — **93%** of it, from a logit lead of three. That is `exp` doing exactly what Chapter 7 promised: turning a modest advantage into a decisive one.

```
logits = [[3.6000001, 0.60000014, -0.30000007]]
probs  = [[0.9345541, 0.04652871, 0.018917156]]
```

## The Book's Last Number

```
prob sum = [[0.99999994]]
```

**Not 1.0.** The probabilities sum to $0.99999994$.

Chapter 7 ran this same check on Burn's built-in `softmax` and got `[1.0]`, exactly. This hand-rolled one does not, and the difference is six parts in a hundred million — three floating-point operations, each rounding, and the error had nowhere to cancel.

It does not matter. It will never matter. But it is the truth about what the machine did.

Run the PyTorch counterpart and you get a *third* answer — `1.0000001192092896`, missing in the other direction. Three mathematically identical computations, three different sums, none of them wrong.

## Gotchas

- **Never compare floats with `==`.** This is why `assert!(sum == 1.0)` is a bug and `assert!((sum - 1.0).abs() < 1e-6)` is not. Chapter 1 introduced `check_closeness` for exactly this reason.
- **A hand-rolled softmax is where `NaN` comes from.** This example calls `exp(logits)` directly, which is fine because the logits are small. On real logits it is a landmine: `exp(1000)` is infinity, and infinity over infinity is `NaN`. The fix is to subtract the row maximum first, since the constant cancels. Burn's `softmax` does it for you; PyTorch's does too. When you write your own, in any framework, you are now the one responsible.

## Key Concepts Covered
- **Softmax From Primitives:** Building normalization out of `exp` / `sum_keepdim` / `broadcast_div`.
- **Keepdim Reductions:** Why the denominator must stay `[1, 1]` rather than collapsing to a scalar, so the divide can broadcast.
- **Floating-Point Honesty:** Reading the output the machine produced instead of the one the math predicted.
- **Numerical Stability:** Knowing what the built-in implementations do for you, and what breaks when you skip it.

## Running It
```bash
cargo run -- -c 15 -e 3 rust
cargo run -- -c 15 -e 3 python
```
