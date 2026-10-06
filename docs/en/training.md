# Training and Saving State

[Documentation](README.md) · [Tensor framework](tensor-framework.md) · [中文](../zh/training.md) | [日本語](../ja/training.md) | [Deutsch](../de/training.md) | [Русский](../ru/training.md)

## Configure the training backend

Use `Autodiff<Cuda<f32, i32>>` to record backward graphs for CUDA tensors, `ruda-nn` to define layers, and `ruda-optim` to update parameters. For NVIDIA environment setup, see [Getting started](getting-started.md).

Add these dependencies to your application's `Cargo.toml`. The example places the application directory alongside the `RUDA` source directory:

```toml
[dependencies]
ruda-autodiff = { path = "../RUDA/ruda-autodiff", default-features = false, features = ["std"] }
ruda-model = { path = "../RUDA/ruda-model", default-features = false, features = ["std"] }
ruda-nn = { path = "../RUDA/ruda-nn", default-features = false, features = ["std"] }
ruda-optim = { path = "../RUDA/ruda-optim", default-features = false, features = ["std"] }
ruda-tensor-device = { path = "../RUDA/ruda-tensor-device", default-features = false, features = ["std", "cuda"] }
```

## Forward, backward, and parameter updates

This training step takes an actual batch `x` and targets `y`, computes mean squared error, and updates a linear layer. For two input features and one output, `x` has shape `[batch, 2]` and `y` has shape `[batch, 1]`. Both use backend `B`, the same device, and F32 data.

```rust
use ruda_autodiff::Autodiff;
use ruda_model::tensor::Tensor;
use ruda_nn::{Linear, LinearConfig};
use ruda_optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor};
use ruda_tensor_device::cuda::{Cuda, CudaDevice};

type B = Autodiff<Cuda<f32, i32>>;
type Model = Linear<B>;
type AdamOptimizer = OptimizerAdaptor<Adam, Model, B>;

fn train_step(
    model: Model,
    optimizer: &mut AdamOptimizer,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
    learning_rate: f64,
) -> Model {
    let residual = model.forward(x) - y;
    let loss = residual.square().mean();
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    optimizer.step(learning_rate, model, gradients)
}
```

Initialize the model and optimizer, then pass them with your batch to `train_step`:

```rust
fn initialize(device: &CudaDevice) -> (Model, AdamOptimizer) {
    let model = LinearConfig::new(2, 1).init::<B>(device);
    let optimizer = AdamConfig::new().init();
    (model, optimizer)
}
```

`backward()` consumes the loss tensor and produces gradients. `GradientsParams::from_grads` associates them with model parameters. Keep the new model returned by each `optimizer.step` call and retain the optimizer between steps to preserve Adam's momentum state.

## Gradient accumulation and learning-rate scheduling

When a full batch does not fit in device memory, run forward and backward for several microbatches before updating parameters once. This function requires equal sample counts in all microbatches. Dividing each mean loss by the number of microbatches produces the gradient of the combined mean loss.

```rust
use ruda_optim::GradientsAccumulator;
use ruda_optim::lr_scheduler::{
    LrScheduler,
    step::{StepLrScheduler, StepLrSchedulerConfig},
};

fn train_window(
    mut model: Model,
    optimizer: &mut AdamOptimizer,
    scheduler: &mut StepLrScheduler,
    batches: &[(Tensor<B, 2>, Tensor<B, 2>)],
) -> Model {
    if batches.is_empty() {
        return model;
    }
    let mut accumulator = GradientsAccumulator::new();
    for (x, y) in batches {
        let residual = model.forward(x.clone()) - y.clone();
        let loss = residual.square().mean() / batches.len() as f64;
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        accumulator.accumulate(&model, gradients);
    }
    model = optimizer.step(scheduler.step(), model, accumulator.grads());
    model
}
```

`accumulate` adds gradients without averaging. `grads()` returns the accumulated gradients and resets the accumulator. For unequal microbatch sizes, weight losses by sample count instead of using the equal-weight division above.

Call `scheduler.step()` once per parameter update, not after each microbatch. Create it with `StepLrSchedulerConfig::new(1e-3, 100).with_gamma(0.5).init()`: the learning rate starts at `1e-3` and is multiplied by `0.5` every 100 calls. Initialization returns `Result<StepLrScheduler, String>`.

## FP32 masters and mixed parameter storage

`Module::to_dtype(FloatDType::F16)` / `to_dtype(FloatDType::BF16)` converts floating parameter storage while retaining IDs, tied aliases and frozen settings. Converted trainable parameters are graph leaves; conversion itself is not differentiated. Use it at model setup or restoration, not as an operation inside an active forward graph.

Wrap an existing `SimpleOptimizer` with `Fp32MasterOptimizer` to retain authoritative FP32 parameters and the original optimizer state. Updates run in FP32 and return parameters in their incoming storage dtype. For the CUDA model defined above:

```rust
use ruda_model::{module::Module, tensor::FloatDType};
use ruda_optim::{AdamW, AdamWConfig, Fp32MasterOptimizer};

fn initialize_master(
    device: &CudaDevice,
) -> (Model, OptimizerAdaptor<Fp32MasterOptimizer<AdamW>, Model, B>) {
    let model = LinearConfig::new(2, 1)
        .init::<B>(device)
        .to_dtype(FloatDType::BF16);
    let optimizer = Fp32MasterOptimizer::new(AdamWConfig::new().build())
        .init::<B, Model>();
    (model, optimizer)
}
```

Match activation storage to the layer's requirements. For FP32 accumulation, replace `accumulator.accumulate(&model, gradients)` with `accumulator.accumulate_with_dtype(&model, gradients, FloatDType::F32)`; incoming and pending gradients are converted before addition. This does not rescale losses, average microbatches or change parameter storage. After restoration, continue using the same accumulation dtype.

`Fp32MasterOptimizer::with_gradient_scale(scale)` explicitly divides gradients by a finite positive loss scale in FP32 before optional `with_grad_clipping`. Scale the loss yourself and keep one scale throughout an accumulation window. Clipping uses RUDA's existing per-parameter rule, not global model-norm clipping; these options do not enable dynamic scaling or automatic nonfinite-step skipping.

## Save and restore training state

`TrainingRecord` saves the model, optimizer, learning-rate scheduler, pending accumulated gradients, and caller state together. These functions reuse the types above and receive the active training state. Saving does not create a new optimizer, scheduler, or accumulator.

```rust
use ruda_model::record::{BinFileRecorder, FullPrecisionSettings, RecorderError};
use ruda_optim::training::{RestoredTraining, TrainingRecord};
use std::path::Path;

type Snapshot = TrainingRecord<B, Model, AdamOptimizer, StepLrScheduler, (usize, usize)>;

fn save_training(
    path: &Path,
    model: &Model,
    optimizer: &AdamOptimizer,
    scheduler: &StepLrScheduler,
    accumulator: &GradientsAccumulator<Model>,
    completed_updates: usize,
    pending_microbatches: usize,
) -> Result<(), RecorderError> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    Snapshot::capture(
        model, optimizer, scheduler, accumulator,
        (completed_updates, pending_microbatches),
    )?.save(&recorder, path.into())
}

fn restore_training(
    path: &Path,
    device: &CudaDevice,
    model: Model,
    optimizer: AdamOptimizer,
    scheduler: StepLrScheduler,
) -> Result<RestoredTraining<Model, AdamOptimizer, StepLrScheduler, (usize, usize)>, RecorderError> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    Snapshot::load(&recorder, path.into(), device)?
        .restore(model, optimizer, scheduler, device)
}
```

Passing `Path::new("checkpoints/step-100")` writes `checkpoints/step-100.bin`. `completed_updates` counts completed parameter updates, and `pending_microbatches` counts accumulated microbatches in the current window. Retrieve both counters from `restored.state` after restoring.

Restore with the same model structure, Adam configuration, and scheduler configuration used when saving. Continue with `restored.accumulator`; do not clear it early or replay microbatches already accumulated. Put data-iteration position and RNG state in caller state `U` and restore them before fetching the next batch. `TrainingRecord` does not automatically snapshot a DataLoader.

For mixed storage, `TrainingRecord::capture_with_dtypes` also captures `ModuleDTypeRecord` and returns `TrainingRecord<B, M, O, S, (ModuleDTypeRecord, U)>`. Load that stored type and call `restore_with_dtypes` to recover each floating parameter's storage dtype while returning the original caller state `U`. Use the actual FP32-master optimizer type for `O` and full-precision recorder settings to preserve master parameters, optimizer moments and pending FP32 gradients. Recreate the same optimizer options, including loss scale and clipping, before restoring its state. Ordinary `capture` / `restore` and their record format remain unchanged.

## Replicated training and differentiable tensor collectives

For explicit rank/device assignment, rendezvous startup, collective order and rank-local checkpoints, follow the [distributed training guide](distributed-training.md).

Enable `ruda-optim/collective` and initialize `DataParallel<B, C>` before constructing optimizer state, using an explicit root and a communicator for `B::InnerBackend`. Replica paths, shapes, dtypes and frozen/tied structure must match; rank-local parameter IDs may differ and are retained. The default `C` is ruCCL's host-staged `RankCommunicator<TensorDevice<B::InnerBackend>>`; a `DataParallelCommunicator` can supply native device transport without changing replica or optimizer semantics.

`initialize` broadcasts floating parameters. `initialize_with_buffers` additionally broadcasts I32/I64 and Bool parameter buffers once; it retains local IDs, integer widths, aliases and frozen flags. Buffers do not enter gradient updates, and this is not automatic synchronization before each forward pass.

Accumulate gradients of **local loss sums**, then call `reduce(&model, gradients, local_weight, policy)` or `reduce_fp32(...)` at the accumulation boundary. `local_weight` is the effective token/sample count for that entire window; the result divides the global gradient sum by the summed count. `reduce` returns the parameter storage dtype; `reduce_fp32` retains FP32 output and accepts FP32 accumulated gradients for half-storage parameters. All ranks must select the same mode and `MissingGradientPolicy`; globally unused parameters remain absent. Call the optimizer and scheduler yourself after reduction, and save each rank's model, optimizer and continuation state together.

`ruda_autodiff::collective` instead connects communication inside the tensor graph:

| API | Forward and backward |
| --- | --- |
| `all_gather` / `all_gather_dim` | Gather equal rank-ordered shards; backward sums and scatters all ranks' gradients |
| `reduce_scatter_sum` / `reduce_scatter_sum_dim` | Sum and scatter equal shards; backward gathers output gradients |
| `reduce_scatter_mean` / `reduce_scatter_mean_dim` | Mean and scatter; backward gathers and divides by world size |
| `all_reduce_sum` / `all_reduce_mean` | Replicated sum/mean; backward sums rank-local gradients, dividing by world size for Mean |
| `broadcast` | Broadcast an explicit root; backward sums into root and gives non-root placeholder inputs zeros |

Unsuffixed gather/scatter use axis zero; `_dim` variants accept positive or negative `AsIndex` axes and retain tensor axis order. Shards must have equal shapes, and the selected scatter axis must divide evenly by world size. All ranks must agree on shapes, dtypes, gradient tracking, broadcast root and collective order in forward and backward. Tracked operations retain the communicator and do not replay communication during checkpoint recomputation. These primitives do not automatically implement ZeRO/FSDP, parameter sharding, tensor parallelism or pipeline scheduling.

See the [ruCCL guide](libraries/ruccl.md) for transport entry points. [rust-ascend](https://github.com/shuqi2077/rust-ascend) reuses these APIs through native NPU `HcclCommunicator` and `Autodiff<RudaAscend>`.

## Choose another optimizer

`ruda_optim` also provides `SgdConfig`, `AdamWConfig`, `AdaGradConfig`, `RmsPropConfig`, `AdanConfig`, `MuonConfig`, and `LBFGSConfig`. When switching optimizers, change both the configuration and state type. See [ruCCL](libraries/ruccl.md) for collective training integration.

API reference: [Optimizers](../../ruda-optim/src/optim/mod.rs), [Training records](../../ruda-optim/src/training.rs).

## Native PyTorch training on `ruda:0`

This is separate from the Rust `Autodiff<Cuda<...>>` examples above. Prepare matching Rust and C++ components from the same source, built locally or from a compatible precompiled bundle: base ABI 10, training API 4, router API 1, paged-backward API 2 and graph API 3. See the [native PyTorch guide](../../ruda-torch/README.md) and [API contracts](native-pytorch-api.md).

`ruda_torch.RMSNorm`/`rms_norm`, `LayerNorm`/`layer_norm` and `silu_mul` provide native first-order training. Normalization is last-axis only; statistics are FP32, outputs retain the activation dtype, and affine parameters may use the input dtype or FP32. Standard `torch.nn.LayerNorm` selects the native training path for supported contiguous last-axis inputs. Higher-order gradients and training graph capture are not supported by these fused interfaces.

`ruda_torch.AdamW(params, fused_step=True, max_grad_norm=1.0, hierarchical_stats=True)` explicitly enables read-only gradient analysis, global unscaled L2 clipping and hierarchical statistics. Both boolean options default to `False`; clipping defaults to `None` and requires `fused_step=True`. The fused path reads one 12-byte report before updates and skips the whole step on nonfinite gradients. It preserves `.grad`, keeps FP32 master parameters/moments, and still launches one update kernel per active parameter, not one GPU kernel for the whole model. Clear gradients before the next accumulation window. On one `ruda_torch.GradScaler` instance, call `scaler.scale(loss).backward()`, `scaler.step(optimizer)` and `scaler.update()` in that order.

Hierarchical statistics use fan-in 1024, at most two extra merge kernels and 49,200 bytes of reusable reduction scratch; this excludes the initial statistics workspace. With at most 1024 rows, no extra merge kernel is needed. This changes reduction order, not the clipping policy.

`PagedAttentionPlan(..., backward_strategy="ordered")` enables atomic-free history gradients; `"atomic"` remains the default. Autograd allocates only requested gradients. See the [ruDNN guide](libraries/rudnn.md) for history compaction and fixed-selection router/expert training, and the [ruBLAS guide](libraries/rublas.md) for grouped backward.

Save model, optimizer and scaler `state_dict()` values plus data position/RNG state together. Optimizer checkpoints preserve `fused_step` and `max_grad_norm`; hierarchical checkpoints carry step-options version 2. Missing options restore with fused/hierarchical modes disabled. [StaticGraph](static-pytorch-graphs.md) supports opt-in first-order training with native forward and same-device recomputed backward; it does not capture optimizer updates. Ordinary model forward/backward uses the separate [AOT entry point](model-compiler.md).

## Adapter and packed-base fine-tuning

The [LoRA/NF4 guide](finetuning.md) covers Rust dense adapters and model-independent native PyTorch adapters, CPU or streamed NF4 preparation, causal labels, full-vocabulary chunked loss and token-weighted accumulation. It also specifies adapter export, exact base/optimizer identity, step-boundary checkpoints and resume semantics; these are different from saving a complete model above.
