# ruda

Unified entry point for Ruda GPU kernels and compute backends. Select CUDA, HIP, or WGPU through Cargo features. The portable host runtime lives in `ruda-runtime` and remains available through `ruda::runtime`.

## Interfaces

- `runtime::client` and `runtime::server`: submit work and implement compute services.
- `runtime::storage`, `memory_management`, and `allocator`: manage backend allocations.
- `runtime::backend`, `compiler`, and `kernel`: implement runtime and compiled-kernel contracts.
- `dsl` and `prelude`: kernel DSL and launch interfaces, enabled by any GPU backend.
- `cuda`, `hip`, and `wgpu`: the selected driver's public API.

## Usage

Cargo package: `ruda`. Rust import: `ruda`.

```bash
cargo add ruda --features cuda
```

For AMD HIP or a cross-vendor WGPU backend, choose one of:

```bash
cargo add ruda --features hip
cargo add ruda --features wgpu
```

CUDA requires a compatible NVIDIA driver and CUDA Toolkit; HIP requires a compatible AMD GPU and ROCm/HIP environment; WGPU requires a compatible GPU and graphics driver. Cargo adds the dependency; it does not install these drivers.

Use `ruda::cuda::{CudaDevice, CudaRuntime}`, `ruda::hip::{AmdDevice, HipRuntime}`, or `ruda::wgpu::{WgpuDevice, WgpuRuntime}` with `ruda::prelude::*`.

From the source workspace, run the same vector-add kernel with the selected backend:

```bash
cargo run -p ruda --example vector-add --features cuda
cargo run -p ruda --example vector-add --features hip
cargo run -p ruda --example vector-add --features wgpu
```

When enabling multiple backends, select one explicitly for the example:

```bash
cargo run -p ruda --example vector-add --features cuda,wgpu -- wgpu
```

## Features

Default features: `runtime-default`.

| Feature | Purpose |
| --- | --- |
| `cuda` | NVIDIA CUDA backend, exposed as `ruda::cuda`. |
| `hip` | AMD ROCm/HIP backend, exposed as `ruda::hip`. |
| `wgpu` | Cross-vendor WGPU backend, exposed as `ruda::wgpu`. |
| `direct-ptx` | Enable CUDA and its direct PTX compiler; select it with `RUDA_CUDA_COMPILER=ptx`. |
| `kernel` | Expose the kernel DSL and prelude without selecting a GPU backend. |
| `runtime` | Expose the runtime modules. |
| `runtime-std` | Enable host configuration and standard-library integration. |
| `runtime-storage-bytes` | Enable byte-backed storage support. |
| `runtime-tracing` | Enable runtime tracing. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/runtime-api.md)
