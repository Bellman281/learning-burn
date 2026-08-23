# Candle Tensors and the `?` Operator (Example 15.1)

An introduction to **Candle**, HuggingFace's minimalist Rust ML framework, and the one design decision that separates it from Burn: shapes are runtime values, not types.

## The Core Concept
Candle is the *other* Rust ML framework. It is smaller than Burn, aimed squarely at **running** pretrained models rather than training new ones, and it is where the Rust ecosystem's Llama and Whisper loaders already live. Its tensor API will feel immediately familiar — `matmul` is `matmul`, and `[2, 3] @ [3, 2]` gives `[2, 2]` exactly as the shape rule says it must.

Then you notice the sigils. Every operation ends in `?`, and `main` returns a `Result`.

## The One Big Difference
- **Burn puts the rank in the type.** `Tensor<B, 2>` and `Tensor<B, 3>` are *different types*, so a whole class of mistake will not compile.
- **Candle carries the shape as data.** A tensor is a tensor; a mismatch comes back as `Err` at run time. Rust will not let you ignore the `Result`, but the compiler cannot catch the error in advance.

Neither is wrong — they are different bets. Burn's pays off when you are **writing** models and inventing shapes you will get wrong. Candle's pays off when you are **loading** a model whose architecture is already fixed and correct: the shapes come from a config file, they were right before you arrived, and encoding them in the type system buys nothing while costing flexibility.

## Key Concepts Covered
- **`Result`-Returning Ops:** Why `let sum = (&a + &b)?;` needs the `?` — shape validation happens at run time and hands back a value you must handle.
- **Explicit Devices:** Candle makes you name the device (`Device::Cpu`; `new_cuda` / `new_metal` for GPU) rather than inferring one.
- **Reading Tensors Back:** `to_vec2::<f32>()` pulls a rank-2 tensor into nested `Vec`s — itself a fallible operation.
- **Familiar Shape Arithmetic:** Confirming that everything learned about matmul and broadcasting in the Burn chapters transfers unchanged.

## Running It
```bash
cargo run -- -c 15 -e 1 rust
cargo run -- -c 15 -e 1 python
```

Both print the same four lines. The difference is not in the numbers — it is that the Rust version had to say `?` four times to get them.
