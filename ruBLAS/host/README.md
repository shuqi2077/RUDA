# rublas-host

CPU matrix multiplication and vector cross products for Ruda host tensors. This is the host implementation used by `ruda-tensor-host`, separate from the GPU `rublas` package.

## Interfaces

- `matmul` and `int_matmul` handle floating-point and integer host matrix products.
- `cross` computes host vector cross products.
- The implementation uses host tensor shapes and strides and selects optional SIMD and parallel paths through features.

## Usage

Cargo package: `rublas-host`. Rust import: `rublas_host`.

```toml
[dependencies]
rublas-host = "0.1"
```

## Features

Default features: `std`, `simd`, `rayon`.

| Feature | Purpose |
| --- | --- |
| `simd` | Enable SIMD support. |
| `rayon` | Enable Rayon and parallel GEMM. |
| `apple-amx` | Enable experimental Apple AMX GEMM. |
| `x86-v4` | Enable the GEMM x86-v4 path. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruBLAS/host/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruBLAS/host/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rublas.md)
