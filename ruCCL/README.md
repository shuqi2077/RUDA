# ruCCL

**English** | [简体中文](docs/zh/README.md)

This repository is a source mirror. Run the commands below from the [RUDA monorepo](https://github.com/shuqi2077/RUDA) root.

Ruda's collective communication library. The public tensor interface reuses Ruda tensor backends and compute libraries; `rank` and `in_process` provide device-independent communication protocols, scheduling, and device-adapter contracts.

## CUDA tensor example

```sh
cargo run -p ruCCL --features cuda --example all_reduce
```

Requires a working NVIDIA driver and CUDA Toolkit. The example creates four logical ranks on GPU 0, performs tensor computation and Ring AllReduce, reads back results, and closes the session. It checks Sum/Mean over 257 FP32 elements and verifies that the original inputs remain unchanged.

## Features

| Feature | Scope |
| --- | --- |
| Default | General communication core and tensor-backend interfaces; does not automatically enable a GPU backend |
| `cuda` | CUDA tensor backend, retaining its default fusion and tuning configuration |
| `test-cuda` | Selects the existing CUDA test backend in addition to `cuda` |
| `test-wgpu` / `test-metal` / `test-vulkan` | Existing WGPU test entry points; run separately from the CUDA test features |
| `tracing` | Existing cross-layer tracing integration |

The public tensor API includes `register`, `all_reduce`, `reduce`, `broadcast`, and `finish_collective`. All ranks must call matching collective operations in the same order. Autodiff callers use the inner backend; the optimizer layer handles gradient synchronization.
