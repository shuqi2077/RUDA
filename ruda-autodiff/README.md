# ruda-autodiff

Automatic differentiation over Ruda tensor backends. The `Autodiff` backend decorator records operations and propagates gradients while retaining the underlying backend's device execution.

## Interfaces

- `Autodiff<B, C>` wraps backend `B` with checkpoint strategy `C`.
- `grads`, `ops`, and `checkpoint` expose gradient, operation, and recomputation interfaces.
- `solver_host` adds explicit first-order host FP64 solver integration when enabled.

## Usage

Cargo package: `ruda-autodiff`. Rust import: `ruda_autodiff`.

```toml
[dependencies]
ruda-autodiff = "0.21"
```

## Features

Default features: `std`, `tracing`.

| Feature | Purpose |
| --- | --- |
| `solver-host` | Enable host numerical-solver differentiation. |
| `distributed` | Enable distributed autodiff integration. |
| `tracing` | Enable autodiff tracing. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-autodiff/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-autodiff/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md)
