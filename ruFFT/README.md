# ruFFT

**English** | [简体中文](docs/zh/README.md)

Fast Fourier transforms for Ruda.

- Cargo package: `ruFFT`
- Rust crate: `rufft`

## Features

| Feature | Operations |
| --- | --- |
| `tensor` | Real-signal FFTs and inverse transforms on device tensors |

Use `rufft::tensor::{rfft, irfft}` to allocate transform outputs, or `rfft_launch` and `irfft_launch` to work with device bindings directly.

## Quick Start

Build from the RUDA workspace:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruFFT --no-default-features --features std,tensor
```

## Documentation

- [User guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rufft.md)
- [Environment setup](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md)
- [Cargo features](Cargo.toml) · [Module exports](src/lib.rs)
