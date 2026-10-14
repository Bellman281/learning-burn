# Bare-Metal Embedded Inference (Example 21.1)

An implementation showcasing how to serialize model weights into memory and bake them directly into a compiled binary for bare-metal systems and microcontrollers without a filesystem.

## The Core Concept
Embedded systems and microcontrollers (such as a Cortex-M4) typically lack operating systems, filesystems, and the memory capacities required for heavy runtimes. While Python-based frameworks structurally cannot deploy to these platforms, **Rust** can compile to target bare metal. 

Instead of searching for files on disk, we serialize weights into a byte buffer and load them directly out of flash memory/RAM.

## How It Works
- **In-Memory Serialization:** The module's parameters are taken as a `ModuleRecord` (`model.into_record()`) and written to an in-memory burnpack buffer with `into_bytes()`; `ModuleRecord::from_bytes` reads them back. Both are in `burn-core` and work without `std`.
- **Baking into Firmware:** On the device, these bytes are embedded straight into the application's executable binary at compile-time using Rust's `include_bytes!` macro. With Burn's `store` feature, `BurnpackStore::from_static(MODEL)` plus `model.load_from(&mut store)` reads the weights from that static slice without copying it first.
- **Zero File I/O:** During runtime, the model deserializes its parameters directly from the program memory space, removing any dependence on disk resources or operating system kernels.
