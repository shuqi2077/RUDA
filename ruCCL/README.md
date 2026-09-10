# ruCCL

**English** | [简体中文](docs/zh/README.md) | [日本語](docs/ja/README.md) | [Deutsch](docs/de/README.md) | [Русский](docs/ru/README.md)

This repository is a source mirror. Run the commands below from the [RUDA monorepo](https://github.com/shuqi2077/RUDA) root.

Ruda's collective communication library. The public tensor interface reuses Ruda tensor backends and compute libraries; `rank` and `in_process` provide device-independent communication protocols, scheduling, and device-adapter contracts.

## CUDA tensor example

```sh
cargo run -p ruCCL --features cuda --example all_reduce
```

Requires a working NVIDIA driver and CUDA Toolkit. The example creates four logical ranks on GPU 0, performs tensor computation and Ring AllReduce, reads back results, and closes the session. It checks Sum/Mean over 257 FP32 elements and verifies that the original inputs remain unchanged.

## Features

| Feature | Scope |
| --- | --- |
| Default | General communication core and tensor-backend interfaces; does not automatically enable a GPU backend |
| `cuda` | CUDA tensor backend, retaining its default fusion and tuning configuration |
| `test-cuda` | Selects the existing CUDA test backend in addition to `cuda` |
| `test-wgpu` / `test-metal` / `test-vulkan` | Existing WGPU test entry points; run separately from the CUDA test features |
| `tracing` | Existing cross-layer tracing integration |

The public tensor API includes `register`, `all_reduce`, `reduce`, `broadcast`, and `finish_collective`. All ranks must call matching collective operations in the same order. Autodiff callers use the inner backend; the optimizer layer handles gradient synchronization.

## ruCCL User Guide

[Compute libraries](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/README.md) · [Tensors and frameworks](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md) · [中文](docs/zh/README.md)

### 1. Layers and entry points

The Cargo package is `ruCCL` and the Rust crate is `ruccl`.

ruCCL includes tensor Backend collectives, a rank core, and in-process implementations. `ruda-communication` provides communication infrastructure. The `orchestrator` feature enables orchestration entry points.

### 2. Tensor collective API

| Function | Behavior |
| --- | --- |
| `register<B>` | Registers a peer, device, and CollectiveConfig |
| `all_reduce<B>` | Returns the reduced result to participants |
| `broadcast<B>` | Sender passes Some(tensor); receivers pass None |
| `reduce<B>` | Reduces to a specified root; non-root participants receive None |
| `finish_collective<B>` | Ends the peer's collective session |
| `reset_collective<B>` | Resets the local collective service and discards registrations and in-progress operation state |

Interfaces use `B: ruda_tensor::Backend` and `B::FloatTensorPrimitive`. When integrating with automatic differentiation, register the inner Backend; a collective call does not itself define an automatic backward rule.

### 3. Registration and call contracts

Create configuration with `CollectiveConfig::default()`. Use `with_num_devices` for the number of local participating devices. Configure strategies and multinode addresses through their configuration methods.

Participants must agree on device counts, use unique peer IDs, and call matching collectives in the same order. Shape, reduction operation, root, and other parameters must agree. Each broadcast must have exactly one sender.

For multinode execution, configure node counts, global and local addresses, and data-service ports together.

### 4. Errors and lifecycle

`CollectiveError` covers duplicate or missing registrations, shape mismatches, inconsistent reduction operations or roots, and invalid broadcast sender counts.

Use `finish_collective` for normal completion. `reset_collective` forgets in-progress state; it does not complete an operation, checkpoint a device task, or provide lossless recovery.

### 5. CUDA example

The `cuda` feature enables the CUDA tensor backend. Run `cargo run --locked -p ruCCL --features cuda --example all_reduce` to execute Ring AllReduce with four logical ranks on GPU 0. It checks Sum/Mean over 257 FP32 elements, input preservation, and session exit.

Device adapters are in [tensor_device](src/tensor_device). For the optimizer interface, see [explicit-rank gradient reduction](https://github.com/shuqi2077/RUDA/blob/main/ruda-optim/src/optim/grads/collective.rs). Transfers include a host-staged path, not zero-copy P2P.

Source: [collective API](src/api.rs), [configuration](src/config.rs), [rank](src/rank/mod.rs), and [in-process implementation](src/in_process/mod.rs).

### 6. Collective training

Enable `collective` in `ruda-optim`. With an explicitly owned rank communicator, convert backward gradients into `GradientsParams`, call `grads.all_reduce_with::<InnerBackend>(&communicator, ReduceOperation::Mean)?`, then pass the returned gradients to `optimizer.step`. Parameter IDs, gradient shapes, dtypes, and call order must match across ranks. For autodiff training, `InnerBackend` is the backend without the `Autodiff` wrapper.

Run the two-rank training example from the source tree:

```powershell
cargo run --locked -p ruda-optim --features collective,cuda --example collective_training -- run ./collective-training-state
cargo run --locked -p ruda-optim --features collective,cuda --example collective_training -- resume ./collective-training-state
```

`run` requires a directory that does not yet exist. It saves each rank's model and optimizer after the first update, then executes the second update. `resume` restores that directory and executes the second update. With CUDA enabled, both logical ranks in this example use the same default device.

See the [collective training example](https://github.com/shuqi2077/RUDA/blob/main/ruda-optim/examples/collective_training.rs) for the complete call sequence. To also save scheduler state and pending accumulated gradients, use `TrainingRecord` from [Training and saving state](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md).
