# ruda-optim

Optimizer updates, gradient handling, clipping, learning-rate schedules, and training-state records for Ruda models. Device-fused optimizer paths and collective gradient synchronization are explicit features.

## Interfaces

- Crate-root optimizer exports provide optimizer configurations and gradient updates.
- `grad_clipping` and `lr_scheduler` handle clipping and schedules.
- `training` combines model, optimizer, scheduler, and accumulated-gradient records.
- `fused_adamw` supplies opt-in AdamW/AMSGrad implementations.

## Usage

Cargo package: `ruda-optim`. Rust import: `ruda_optim`.

```toml
[dependencies]
ruda-optim = "0.21"
```

## Features

Default features: `std`, `ruda-model/default`.

| Feature | Purpose |
| --- | --- |
| `fused-adamw` | Enable host fused-optimizer interfaces. |
| `fused-adamw-device` | Enable the runtime-generic device implementation. |
| `fused-adamw-cuda` | Enable CUDA fused-optimizer integration. |
| `collective` | Enable ruCCL gradient synchronization. |
| `gradient-guard` | Enable the gradient-guard integration. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-optim/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-optim/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md)
