# ruda-optim

Optimizer updates, gradient handling, clipping, learning-rate schedules, and training-state records for Ruda models. Device-fused optimizer paths and collective gradient synchronization are explicit features.

## Interfaces

- Crate-root optimizer exports provide optimizer configurations and gradient updates.
- `grad_clipping` and `lr_scheduler` handle clipping and schedules.
- `training` combines model, optimizer, scheduler, and accumulated-gradient records.
- `Fp32MasterOptimizer::new(existing_optimizer).init()` keeps authoritative FP32 parameters and the wrapped optimizer's state while returning parameters in their original F32/F16/BF16 storage dtype. `with_gradient_scale` explicitly unscales in FP32 before optional per-parameter `with_grad_clipping`; neither option enables dynamic scaling or automatic step skipping.
- `GradientsAccumulator::accumulate_with_dtype(&model, gradients, FloatDType::F32)` converts incoming and pending gradients before addition without changing parameter storage or loss normalization.
- `TrainingRecord::capture_with_dtypes` records per-parameter floating storage metadata with caller state; load the resulting record type and use `restore_with_dtypes` to restore mixed storage, FP32 masters and pending gradients together. Use full-precision recorder settings for FP32 state.
- With `collective`, `data_parallel::DataParallel::initialize` validates replica paths, shapes, dtypes and tied aliases, then broadcasts floating parameters from an explicit root. Local parameter IDs are preserved.
- `DataParallel::initialize_with_buffers` additionally broadcasts I32/I64 and Bool parameter buffers once. Buffers retain IDs, widths and aliases and do not enter gradient updates; this does not enable per-forward buffer synchronization.
- `DataParallel::reduce` synchronizes gradients of local loss sums and divides by the total effective token/sample count. `reduce_fp32` retains FP32 output for half-storage parameters and accepts FP32 accumulation; all ranks must select the same reduction mode. Accumulate locally before reducing; `MissingGradientPolicy` explicitly selects rejection or zero contribution for unused parameters. Globally unused parameters remain absent. The default transport is ruCCL host-staged; `DataParallel<B, C>` accepts a `DataParallelCommunicator<B::InnerBackend>` such as rust-ascend's native HCCL adapter. This is replicated data parallelism, not TP/PP/FSDP. Save each rank's model/optimizer/continuation records together.
- `fused_adamw` supplies opt-in AdamW/AMSGrad implementations.
- `fused_adamw::storage` supplies preallocated native-adapter kernels under `fused-adamw-device`; `stats_plan::StatsPlan` plans bounded hierarchical gradient reductions. The out-of-place `adamw_step` API and default Rust optimizer remain separate. See the [fused AdamW guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/fused-adamw.md) and [native PyTorch training](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md).

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
