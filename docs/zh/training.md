# 训练与状态保存

[文档首页](README.md) · [张量框架](tensor-framework.md) · [English](../en/training.md) | [日本語](../ja/training.md) | [Deutsch](../de/training.md) | [Русский](../ru/training.md)

## 配置训练后端

使用 `Autodiff<Cuda<f32, i32>>` 为 CUDA 张量记录反向图，使用 `ruda-nn` 定义网络层，使用 `ruda-optim` 更新参数。NVIDIA 环境配置见[快速开始](getting-started.md)。

在应用的 `Cargo.toml` 中加入以下依赖。示例中，应用目录与 `RUDA` 源码目录同级：

```toml
[dependencies]
ruda-autodiff = { path = "../RUDA/ruda-autodiff", default-features = false, features = ["std"] }
ruda-model = { path = "../RUDA/ruda-model", default-features = false, features = ["std"] }
ruda-nn = { path = "../RUDA/ruda-nn", default-features = false, features = ["std"] }
ruda-optim = { path = "../RUDA/ruda-optim", default-features = false, features = ["std"] }
ruda-tensor-device = { path = "../RUDA/ruda-tensor-device", default-features = false, features = ["std", "cuda"] }
```

## 前向、反向与参数更新

下面的训练步接收实际批次 `x` 和目标 `y`，计算均方误差并更新一个线性层。对于两维输入、一维输出，`x` 为 `[batch, 2]`，`y` 为 `[batch, 1]`，两者都使用后端 `B`、同一设备和 F32 数据。

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

初始化模型和优化器，再将它们与批次传给 `train_step`：

```rust
fn initialize(device: &CudaDevice) -> (Model, AdamOptimizer) {
    let model = LinearConfig::new(2, 1).init::<B>(device);
    let optimizer = AdamConfig::new().init();
    (model, optimizer)
}
```

`backward()` 消耗损失张量并生成梯度；`GradientsParams::from_grads` 将梯度关联到模型参数。每次更新都要接收 `optimizer.step` 返回的新模型，优化器对象则保留到下一步，以延续 Adam 的动量状态。

## 梯度累积与学习率调度

显存放不下一个完整批次时，可以对多个微批次分别前向和反向，再更新一次参数。下面的函数要求每个微批次含有相同数量的样本；每个微批次的平均损失除以累积次数，使最终梯度对应这些样本的平均损失。

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

`accumulate` 相加梯度，不自动取平均；`grads()` 取出累计梯度并清空累积器。不同大小的微批次应按样本数加权损失，而不是直接套用上面的等权除法。

`scheduler.step()` 在一次参数更新时调用一次，不在每个微批次后调用。使用 `StepLrSchedulerConfig::new(1e-3, 100).with_gamma(0.5).init()` 创建调度器：初始学习率为 `1e-3`，每 100 次调用乘以 `0.5`；初始化返回 `Result<StepLrScheduler, String>`。

## 保存和恢复训练状态

`TrainingRecord` 一起保存模型、优化器、学习率调度器、尚未更新的累计梯度和调用方状态。以下函数复用上面的类型定义，接收正在训练的状态；保存时不创建新的优化器、调度器或累积器。

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

传入 `Path::new("checkpoints/step-100")` 时，记录写入 `checkpoints/step-100.bin`。`completed_updates` 是已经完成的更新数，`pending_microbatches` 是当前窗口已经累计的微批次数；恢复后从 `restored.state` 取回这两个计数。

恢复时使用与保存时相同的模型结构、Adam 配置和调度器配置。继续累积时使用 `restored.accumulator`，不要提前清空它或重复执行已经累计的微批次。数据迭代位置和随机数状态由应用放入调用方状态 `U`，并在取下一批数据前恢复；`TrainingRecord` 不会自动保存 DataLoader。

## 选择其他优化器

`ruda_optim` 还提供 `SgdConfig`、`AdamWConfig`、`AdaGradConfig`、`RmsPropConfig`、`AdanConfig`、`MuonConfig` 和 `LBFGSConfig`。更换优化器时同时更换配置和状态类型。集合通信训练的接入见 [ruCCL](libraries/ruccl.md)。

接口参考：[优化器](../../ruda-optim/src/optim/mod.rs)、[训练记录](../../ruda-optim/src/training.rs)。
