# ruda-nn

Neural-network layers, activation modules, padding, and loss functions for Ruda models. Layers operate through tensor backend contracts rather than selecting a GPU driver themselves.

## Interfaces

- `modules` contains neural-network layer implementations and is re-exported at the crate root.
- `activation` exposes activation modules; `loss` exposes loss functions.
- `Initializer` is re-exported from `ruda-model` for parameter initialization.

## Usage

Cargo package: `ruda-nn`. Rust import: `ruda_nn`.

```toml
[dependencies]
ruda-nn = "0.21"
```

## Features

Default features: `std`, `ruda-model/default`.

| Feature | Purpose |
| --- | --- |
| `std` | Enable standard-library model integration. |
| `sparse` | Enable sparse layer support. |
| `cuda` | Enable the CUDA tensor dependency. |
| `fusion` | Enable device-fusion integration. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-nn/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-nn/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md)
