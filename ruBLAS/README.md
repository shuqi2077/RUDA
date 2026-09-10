# ruBLAS

**English** | [简体中文](docs/zh/README.md)

Linear algebra and matrix multiplication for Ruda.

- Cargo package: `rublas`
- Rust crate: `rublas`

## Features

| Feature | Operations |
| --- | --- |
| `tensor-matmul` | Matrix multiplication |
| `tensor-matmul-autotune` | Matrix multiplication autotuning |
| `tensor-vector` | Vector operations |
| `tensor-grouped` | Grouped matrix multiplication |
| `tensor-int4` | INT4 weight operations |

## Quick Start

Build from the RUDA workspace:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p rublas --features tensor-matmul
```

## Documentation

- [User guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rublas.md)
- [Environment setup](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md)
- [Cargo features](Cargo.toml) · [Module exports](src/lib.rs)
