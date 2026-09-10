# ruPRIM

**English** | [简体中文](docs/zh/README.md)

Parallel primitives, reductions, scans, and indexing for Ruda.

- Cargo package: `ruPRIM`
- Rust crate: `ruprim`

## Features

| Feature | Operations |
| --- | --- |
| `tensor-reduce` | Whole-tensor and axis reductions |
| `tensor-reduce-autotune` | Reduction autotuning |
| `tensor-scan` | Cumulative sum, product, minimum, and maximum |
| `elementwise` | Elementwise operations |
| `indexing` | Selection, slicing, gather, and scatter |

## Quick Start

Build from the RUDA workspace:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruPRIM --features tensor-reduce,tensor-scan,indexing
```

## Documentation

- [User guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/ruprim.md)
- [Environment setup](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md)
- [Cargo features](Cargo.toml) · [Module exports](src/lib.rs)
