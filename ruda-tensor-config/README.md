# ruda-tensor-config

Shared configuration for Ruda tensor fusion and automatic differentiation. It provides the configuration types and loading interface, not tensor computation.

## Interfaces

- `RudaTensorConfig` contains `FusionConfig` and `AutodiffConfig`.
- `RuntimeConfig` is re-exported for loading and accessing configuration.
- Configuration file names are `ruda-tensor.toml` and `Ruda-Tensor.toml`; standard-library builds support environment overrides such as `RUDA_FUSION_LOG`.

## Usage

Cargo package: `ruda-tensor-config`. Rust import: `ruda_tensor_config`.

```toml
[dependencies]
ruda-tensor-config = "0.21"
```

## Features

Default features: `std`.

| Feature | Purpose |
| --- | --- |
| `std` | Enable host configuration integration. |
| `tracing` | Enable tracing-related configuration support. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-tensor-config/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-tensor-config/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md)
