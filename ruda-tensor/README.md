# ruda-tensor

Shared tensor backend contracts, primitives, shapes, data types, and the feature-selected public tensor API. Backend implementations live in the separate host, device, router, and remote packages.

## Interfaces

- `Backend` and the `ops` traits define backend operation contracts.
- `TensorData`, shape, slice, and quantization types describe tensor values and layout.
- `api::Tensor` is the high-level tensor interface when `api` is enabled; `graph` exposes operation graphs.

## Usage

Cargo package: `ruda-tensor`. Rust import: `ruda_tensor`.

```toml
[dependencies]
ruda-tensor = { version = "0.21", features = ["api"] }
```

## Features

Default features: `std`.

| Feature | Purpose |
| --- | --- |
| `api` | Enable the public tensor API. |
| `api-std` | Enable standard-library support for the public API. |
| `graph` | Enable tensor operation graphs. |
| `distributed` | Enable distributed tensor contracts. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-tensor/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-tensor/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md)
