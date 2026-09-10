# ruDNN

**English** | [简体中文](docs/zh/README.md)

Neural network operators for Ruda.

- Cargo package: `ruDNN`
- Rust crate: `rudnn`

## Features

| Feature | Operations |
| --- | --- |
| `tensor-attention` | Attention |
| `tensor-convolution` | Convolution |
| `tensor-normalization` | LayerNorm, RMSNorm, and softmax |
| `tensor-moe` | MoE routing, dispatch, and expert computation |
| `tensor-gated-delta` | Gated-delta computation |
| `pooling` | Pooling |
| `interpolation` | Interpolation |
| `grid-sample` | Grid sampling |
| `ctc` | CTC loss |

## Quick Start

Build from the RUDA workspace:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruDNN --features tensor-normalization
```

## Documentation

- [User guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rudnn.md)
- [Environment setup](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md)
- [Cargo features](Cargo.toml) · [Module exports](src/lib.rs)
