# ruda-fusion

Tensor-operation fusion planning and execution for compatible Ruda backends. It records operation streams, searches for applicable optimizations, and executes fused or unfused operations.

## Interfaces

- `Fusion<B>` decorates a backend implementing `FusionBackend`.
- `stream`, `OperationFuser`, and `Optimization` provide operation-stream and optimization contracts.
- `device` adds device code generation and domain optimizations when enabled.

## Usage

Cargo package: `ruda-fusion`. Rust import: `ruda_fusion`.

```toml
[dependencies]
ruda-fusion = "0.21"
```

## Features

Default features: `std`, `tracing`.

| Feature | Purpose |
| --- | --- |
| `device` | Enable device fusion implementations. |
| `device-tensor` | Enable device tensor integration. |
| `device-autotune` | Enable device fusion autotuning. |
| `distributed` | Enable distributed fusion contracts. |
| `memory-checks` | Enable fusion memory checks. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-fusion/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-fusion/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md)
