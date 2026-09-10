# ruda-tensor-host

CPU tensor backend implemented with the Ruda host-domain libraries. It implements framework tensor operations using host storage, strided layouts, optional SIMD, and optional Rayon parallelism.

## Interfaces

- `Host` and `HostDevice` are the framework backend and device.
- `HostTensor`, `HostQTensor`, and `Layout` represent host tensors and layouts.
- Use `Host` as the backend parameter of `ruda_tensor::api::Tensor`; enable the tensor package's `api` feature in the application.

## Usage

Cargo package: `ruda-tensor-host`. Rust import: `ruda_tensor_host`.

```toml
[dependencies]
ruda-tensor-host = "0.21"
```

## Features

Default features: `std`, `simd`, `rayon`.

| Feature | Purpose |
| --- | --- |
| `simd` | Enable SIMD implementations. |
| `rayon` | Enable Rayon parallel execution. |
| `sparse` | Enable host sparse operations. |
| `critical-section` | Support targets that require critical-section-backed synchronization. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-tensor-host/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-tensor-host/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md)
