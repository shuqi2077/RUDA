# Training and Saving State

[Documentation](README.md) · [Tensor framework](tensor-framework.md) · [中文](../zh/training.md)

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

## Choose another optimizer

`ruda_optim` also provides `SgdConfig`, `AdamWConfig`, `AdaGradConfig`, `RmsPropConfig`, `AdanConfig`, `MuonConfig`, and `LBFGSConfig`. When switching optimizers, change both the configuration and state type. See [ruCCL](libraries/ruccl.md) for collective training integration.

API reference: [Optimizers](../../ruda-optim/src/optim/mod.rs), [Training records](../../ruda-optim/src/training.rs).
