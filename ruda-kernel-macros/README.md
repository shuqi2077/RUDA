# ruda-kernel-macros

Procedural macros that expand Rust kernel functions, traits, and implementations into Ruda IR construction code. The kernel DSL uses this crate during compilation; device execution is handled by a runtime and driver.

## Interfaces

- `#[ruda]` expands kernel functions and supported trait/implementation items.
- `#[ruda(launch)]` generates a checked launch entry point.
- `launch_unchecked` requests an unchecked launch entry point and retains its caller-side obligations.

## Usage

Cargo package: `ruda-kernel-macros`. Rust import: `ruda_kernel_macros`.

```toml
[dependencies]
ruda-kernel-macros = "0.1"
```

## Features

Default features: `kernel-ir`, `std`.

| Feature | Purpose |
| --- | --- |
| `kernel-ir` | Enable the Rust-to-IR macro implementation. |
| `debug_symbols` | Enable the debug-symbol macro feature (also enables `kernel-ir`). |
| `tracing` | Enable tracing integration for IR expansion. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-kernel-macros/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-kernel-macros/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/programming-guide.md)
