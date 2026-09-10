# ruRAND

**English** | [简体中文](docs/zh/README.md)

Random number generation for Ruda.

- Cargo package: `ruRAND`
- Rust crate: `rurand`

## Features

| Feature | Operations |
| --- | --- |
| `tensor` | Uniform, normal, and Bernoulli distributions on device tensors |

Use `rurand::tensor::{random_uniform, random_normal, random_bernoulli}` to allocate tensors, or the functions at the crate root to fill existing device storage. `rurand::seed` sets the shared random seed.

## Quick Start

Build from the RUDA workspace:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruRAND --no-default-features --features std,tensor
```

## Documentation

- [User guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rurand.md)
- [Environment setup](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md)
- [Cargo features](Cargo.toml) · [Module exports](src/lib.rs)
