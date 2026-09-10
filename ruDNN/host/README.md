# ruDNN-host

CPU neural-network operators for Ruda host tensors. It supplies the host backend's attention, activations, convolution, pooling, interpolation, grid sampling, and embedding operations.

## Interfaces

- `activation` and `attention` implement host neural-network operations.
- `convolution`, `pool`, `interpolate`, and `grid_sample` implement spatial operators.
- `embedding` implements host embedding operations.

## Usage

Cargo package: `ruDNN-host`. Rust import: `rudnn_host`.

```toml
[dependencies]
ruDNN-host = "0.1"
```

## Features

Default features: `std`, `simd`, `rayon`.

| Feature | Purpose |
| --- | --- |
| `simd` | Enable SIMD and host-primitives SIMD support. |
| `rayon` | Enable parallel host kernels and GEMM. |
| `apple-amx` | Enable experimental Apple AMX GEMM. |
| `x86-v4` | Enable the GEMM x86-v4 path. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruDNN/host/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruDNN/host/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rudnn.md)
