# ruda-autodiff

Automatic differentiation over Ruda tensor backends. The `Autodiff` backend decorator records operations and propagates gradients while retaining the underlying backend's device execution.

## Interfaces

- `Autodiff<B, C>` wraps backend `B` with checkpoint strategy `C`.
- `grads`, `ops`, and `checkpoint` expose gradient, operation, and recomputation interfaces.
- `solver_host` adds explicit first-order host FP64 solver integration when enabled.
- `collective` adds differentiable tensor operations over an explicitly owned rank communicator, using the underlying backend for transport and shared graph rules for backward.

## Differentiable rank collectives

The functions in `ruda_autodiff::collective` accept `Tensor<Autodiff<B, S>, D>` and a communicator implementing the contracts in `ruda_tensor::collective`.

| Function | Forward | Backward |
| --- | --- | --- |
| `all_gather` / `all_gather_dim` | Gather equal shards in rank order | Sum gradients across ranks and scatter the input shard |
| `reduce_scatter_sum` / `reduce_scatter_sum_dim` | Sum across ranks and return an equal shard | Gather rank-local output gradients |
| `reduce_scatter_mean` / `reduce_scatter_mean_dim` | Average across ranks and return an equal shard | Gather gradients and divide by world size |
| `all_reduce_sum` / `all_reduce_mean` | Sum/average corresponding replicated elements | Sum rank-local gradients, dividing by world size for Mean |
| `broadcast` | Replicate the explicit root's input | Sum all rank-local gradients into root; non-root placeholder inputs receive zeros |

The unsuffixed gather/scatter functions use axis zero; `_dim` variants accept `AsIndex`, including negative axes. Scatter requires the selected axis to be divisible by world size. All ranks must use matching shapes, dtypes, gradient tracking and collective order, and the same broadcast root. The communicator is retained by tracked graph nodes; communication is not replayed by checkpoint recomputation. These functions do not automatically enable parameter sharding or choose a distributed training strategy.

ruCCL's `RankCommunicator<TensorDevice<B>>` implements the transport contracts; rust-ascend's `HcclCommunicator` implements them on native NPU memory. See the [training guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md) for integration with gradient accumulation and optimizers.

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
