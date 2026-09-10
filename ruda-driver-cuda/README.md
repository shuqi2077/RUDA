# ruda-driver-cuda

NVIDIA CUDA execution backend for the Ruda runtime. It connects kernel compilation, CUDA device resources, memory, and execution queues through `CudaRuntime`.

## Interfaces

- `CudaDevice` selects a CUDA device; `CudaRuntime` implements the runtime.
- `install::cuda_path`, `include_path`, and `cccl_include_path` locate CUDA installation files.
- CUDA C++/NVRTC is the default compiler path. Direct PTX is selected explicitly with `direct-ptx` and `RUDA_CUDA_COMPILER=ptx`.

## Usage

Cargo package: `ruda-driver-cuda`. Rust import: `ruda_driver_cuda`.

```toml
[dependencies]
ruda-driver-cuda = "0.1"
```

## Features

Default features: `std`, `ruda/runtime-default`, `ruda-core/std`, `ruda-kernel/frontend-default`.

| Feature | Purpose |
| --- | --- |
| `direct-ptx` | Enable the direct PTX compiler path. |
| `ptx-wmma` | Select the PTX WMMA compiler implementation. |
| `tracing` | Enable cross-layer execution tracing. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-driver-cuda/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-driver-cuda/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/driver-api.md)
