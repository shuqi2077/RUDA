# ruda-driver-cpu

CPU execution backend for Ruda kernels using the MLIR compilation path. This is a compiled-kernel runtime, distinct from the host tensor operators in `ruda-tensor-host`.

## Interfaces

- `CpuDevice` and `CpuRuntime` provide the device and runtime entry points.
- `compilation`, `execution`, and `memory` implement CPU kernel preparation and execution.
- The backend depends on `tracel-llvm` and the compiler's `mlir` feature; its LLVM/MLIR toolchain is required for builds.

## Usage

Cargo package: `ruda-driver-cpu`. Rust import: `ruda_driver_cpu`.

```toml
[dependencies]
ruda-driver-cpu = "0.1"
```

## Features

Default features: `std`, `ruda/runtime-default`, `ruda-core/std`, `ruda-kernel/frontend-default`.

| Feature | Purpose |
| --- | --- |
| `mlir-dump` | Enable compiler MLIR dumps. |
| `std` | Enable standard-library runtime integration. |
| `tracing` | Enable runtime and optimizer tracing. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-driver-cpu/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-driver-cpu/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/driver-api.md)
