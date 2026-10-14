# Serialization Precision and File Overhead (Example 19.2)

A practical investigation into serialization precision in **Burn** and **PyTorch**, highlighting the trade-offs between float precision ($f32$ vs $f16$), inference accuracy, and actual file sizes on disk.

## The Core Concept
A burnpack file stores each weight in the dtype the model holds (here `f32`). An adapter on a `burn-store` `BurnpackStore` changes that on the way out:
- **Full Precision (`f32`):** `BurnpackStore` alone. Lossless; predictions match the trained state bit-for-bit.
- **Half Precision (`f16`):** `BurnpackStore` with `HalfPrecisionAdapter` (`with_to_adapter` on save, `with_from_adapter` on load). Weights are rounded to `f16` on disk and widened back to `f32` on load, causing minor differences in downstream inference values.

Loaded *without* the adapter, the weights keep the file's dtype (`F16`); `ModuleRecord::cast_to_module_dtype()` asks for the model's dtype instead.

## The Format Overhead Paradox
While $f16$ is widely known to halve model size, saving a tiny model (2 parameters) in $f16$ saves just 2 bytes here: 516 bytes vs 514.

Burnpack writes a 10-byte header and the metadata (padded to 256 bytes), then starts every tensor on a 256-byte boundary so the file can be memory-mapped. For microscopic models, header and padding are the file. Once tensors outgrow a 256-byte block (more than 64 values), the weight savings show, and at scale the file is halved.

## Why Use $f16$?
The primary benefit of $f16$ is not disk space, but rather **hardware performance**. By halving the memory footprint, it reduces memory bandwidth bottlenecks during GPU inference, speeding up data transfers to the execution units.
