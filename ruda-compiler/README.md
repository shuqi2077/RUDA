# ruda-compiler

Compiler backends and IR optimization for Ruda kernels. This crate emits target code; it is separate from the driver packages that load and execute that code.

## Interfaces

- `ptx`: direct kernel-IR-to-PTX compilation with explicit target options.
- `cpp`: CUDA C++ and Metal C++ code generation.
- `optimizer`, `wgsl`, `spirv`, and `mlir`: optimization and feature-selected target backends.

## Usage

Cargo package: `ruda-compiler`. Rust import: `ruda_compiler`.

```toml
[dependencies]
ruda-compiler = "0.1"
```

## Features

Default features: `ptx`, `cpp-default`.

| Feature | Purpose |
| --- | --- |
| `ptx` | Enable direct PTX generation. |
| `cpp` | Enable C++ code generation. |
| `wgsl` | Enable WGSL generation. |
| `spirv` | Enable SPIR-V generation. |
| `mlir` | Enable the MLIR backend and its LLVM dependencies. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-compiler/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-compiler/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/compiler-guide.md)
