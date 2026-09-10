# Обучение и сохранение состояния

[English](../en/training.md) | [简体中文](../zh/training.md) | [日本語](../ja/training.md) | [Deutsch](../de/training.md) | **Русский**

[Документация](README.md) · [Тензорная платформа](tensor-framework.md) · [中文](../zh/training.md)

## Настройте бэкенд обучения

Используйте `Autodiff<Cuda<f32, i32>>` для записи графов обратного прохода CUDA-тензоров, `ruda-nn` для определения слоёв и `ruda-optim` для обновления параметров. Настройка NVIDIA описана в разделе [начала работы](getting-started.md).

Добавьте эти зависимости в `Cargo.toml` вашего приложения. В примере каталог приложения размещается рядом с исходным каталогом `RUDA`:

```toml
[dependencies]
ruda-autodiff = { path = "../RUDA/ruda-autodiff", default-features = false, features = ["std"] }
ruda-model = { path = "../RUDA/ruda-model", default-features = false, features = ["std"] }
ruda-nn = { path = "../RUDA/ruda-nn", default-features = false, features = ["std"] }
ruda-optim = { path = "../RUDA/ruda-optim", default-features = false, features = ["std"] }
ruda-tensor-device = { path = "../RUDA/ruda-tensor-device", default-features = false, features = ["std", "cuda"] }
```

## Прямое, обратное обновление и обновление параметров.

На этом этапе обучения берется реальный пакет `x` и нацеливается на `y`, вычисляется среднеквадратическая ошибка и обновляется линейный слой. Для двух входных функций и одного выходного `x` имеет форму `[batch, 2]`, а `y` имеет форму `[batch, 1]`. Оба используют бэкенд `B`, одно и то же устройство и данные F32.

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

Инициализируйте модель и оптимизатор, затем передайте их вместе с пакетом в `train_step`:

```rust
fn initialize(device: &CudaDevice) -> (Model, AdamOptimizer) {
    let model = LinearConfig::new(2, 1).init::<B>(device);
    let optimizer = AdamConfig::new().init();
    (model, optimizer)
}
```

`backward()` использует тензор потерь и создает градиенты. `GradientsParams::from_grads` связывает их с параметрами модели. Сохраняйте новую модель, возвращаемую каждым вызовом `optimizer.step`, и сохраняйте оптимизатор между шагами, чтобы сохранить состояние импульса Адама.

## Накопление градиента и планирование скорости обучения

Если полная партия не помещается в память устройства, выполните несколько микропартий вперед и назад, прежде чем обновить параметры один раз. Эта функция требует одинакового количества образцов во всех микропартиях. Разделив каждую среднюю потерю на количество микропартий, получим градиент совокупной средней потери.

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

`accumulate` добавляет градиенты без усреднения. `grads()` возвращает накопленные градиенты и сбрасывает аккумулятор. Для неравных размеров микропартий потери веса определяются количеством проб вместо использования разделения по равному весу, описанного выше.

Вызовите `scheduler.step()` один раз при каждом обновлении параметра, а не после каждого микропакета. Создайте его с помощью `StepLrSchedulerConfig::new(1e-3, 100).with_gamma(0.5).init()`: скорость обучения начинается с `1e-3` и умножается на `0.5` каждые 100 вызовов. Инициализация возвращает `Result<StepLrScheduler, String>`.

## Сохранить и восстановить состояние тренировки

`TrainingRecord` сохраняет вместе модель, оптимизатор, планировщик скорости обучения, ожидающие накопленные градиенты и состояние вызывающего объекта. Эти функции повторно используют приведенные выше типы и получают активное состояние обучения. При сохранении не создается новый оптимизатор, планировщик или аккумулятор.

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

При передаче `Path::new("checkpoints/step-100")` записывается `checkpoints/step-100.bin`. `completed_updates` подсчитывает завершенные обновления параметров, а `pending_microbatches` подсчитывает накопленные микропартии в текущем окне. Получите оба счетчика из `restored.state` после восстановления.

Восстановление с той же структурой модели, конфигурацией Адама и конфигурацией планировщика, которые использовались при сохранении. Продолжите с `restored.accumulator`; не очищайте его раньше времени и не воспроизводите уже накопленные микропартии. Поместите позицию итерации данных и состояние RNG в состояние вызывающего абонента `U` и восстановите их перед получением следующего пакета. `TrainingRecord` не создает автоматически снимок DataLoader.

## Выбрать другой оптимизатор

`ruda_optim` также предоставляет `SgdConfig`, `AdamWConfig`, `AdaGradConfig`, `RmsPropConfig`, `AdanConfig`, `MuonConfig` и `LBFGSConfig`. При переключении оптимизаторов измените как конфигурацию, так и тип состояния. См. [ruCCL](libraries/ruccl.md) для интеграции коллективного обучения.

API Ссылка: [Оптимизаторы](../../ruda-optim/src/optim/mod.rs), [Записи обучения](../../ruda-optim/src/training.rs).
