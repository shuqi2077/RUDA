# ruda-model

Model configuration, modules, parameters, records, and data loading for Ruda. Neural-network layer implementations are in `ruda-nn`; storage-format adapters are in `ruda-store`.

## Interfaces

- `config`, `module`, and `record` define model configuration, parameter ownership, and recording.
- `Tensor` is re-exported from the model tensor interface.
- `data` provides standard-library data-loading support; optional dataset integration comes from `ruda-dataset`.

## Usage

Cargo package: `ruda-model`. Rust import: `ruda_model`.

```toml
[dependencies]
ruda-model = "0.21"
```

## Features

Default features: `std`, `ruda-core/std`, `ruda-core/id-tests-std`, `ruda-tensor-config/std`, `ruda-dataset?/default`, `ruda-tensor/api-std`.

| Feature | Purpose |
| --- | --- |
| `dataset` | Enable dataset integration. |
| `network` | Enable model I/O dependency integration. |
| `record-item-custom-serde` | Enable custom record-item serialization support. |
| `distributed` | Enable distributed model and tensor integration. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-model/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-model/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md)
