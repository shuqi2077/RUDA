# Trainings- und Speicherstatus

[English](../en/training.md) | [简体中文](../zh/training.md) | [日本語](../ja/training.md) | **Deutsch** | [Русский](../ru/training.md)

[Dokumentation](README.md) · [Tensor-Framework](tensor-framework.md) · [中文](../zh/training.md)

## Konfigurieren Sie das Trainings-Backend

Verwenden Sie `Autodiff<Cuda<f32, i32>>` zum Aufzeichnen von Rückwärtsgraphen für CUDA-Tensoren, `ruda-nn` zum Definieren von Schichten und `ruda-optim` zum Aktualisieren von Parametern. Die NVIDIA-Umgebung beschreibt [Erste Schritte](getting-started.md).

Fügen Sie diese Abhängigkeiten zum `Cargo.toml` Ihrer Anwendung hinzu. Das Beispiel platziert das Anwendungsverzeichnis neben dem Quellverzeichnis `RUDA`:

```toml
[dependencies]
ruda-autodiff = { path = "../RUDA/ruda-autodiff", default-features = false, features = ["std"] }
ruda-model = { path = "../RUDA/ruda-model", default-features = false, features = ["std"] }
ruda-nn = { path = "../RUDA/ruda-nn", default-features = false, features = ["std"] }
ruda-optim = { path = "../RUDA/ruda-optim", default-features = false, features = ["std"] }
ruda-tensor-device = { path = "../RUDA/ruda-tensor-device", default-features = false, features = ["std", "cuda"] }
```

## Vorwärts-, Rückwärts- und Parameteraktualisierungen

Dieser Trainingsschritt nimmt einen tatsächlichen Stapel `x` und zielt auf `y` ab, berechnet den mittleren quadratischen Fehler und aktualisiert eine lineare Ebene. Für zwei Eingabe-Features und eine Ausgabe hat `x` die Form `[batch, 2]` und `y` die Form `[batch, 1]`. Beide verwenden das Backend `B`, dasselbe Gerät und F32-Daten.

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

Initialisieren Sie das Modell und den Optimierer und übergeben Sie sie dann mit Ihrem Stapel an `train_step`:

```rust
fn initialize(device: &CudaDevice) -> (Model, AdamOptimizer) {
    let model = LinearConfig::new(2, 1).init::<B>(device);
    let optimizer = AdamConfig::new().init();
    (model, optimizer)
}
```

`backward()` verbraucht den Verlusttensor und erzeugt Gradienten. `GradientsParams::from_grads` verknüpft sie mit Modellparametern. Behalten Sie das neue Modell bei, das bei jedem `optimizer.step`-Aufruf zurückgegeben wird, und behalten Sie den Optimierer zwischen den Schritten bei, um Adams Impulszustand beizubehalten.

## Gradientenakkumulation und Lernratenplanung

Wenn ein vollständiger Batch nicht in den Gerätespeicher passt, führen Sie mehrere Mikrobatches vorwärts und rückwärts durch, bevor Sie die Parameter einmal aktualisieren. Diese Funktion erfordert gleiche Probenzahlen in allen Mikrobatches. Die Division jedes mittleren Verlusts durch die Anzahl der Mikrochargen ergibt den Gradienten des kombinierten mittleren Verlusts.

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

`accumulate` fügt Gradienten ohne Mittelwertbildung hinzu. `grads()` gibt die akkumulierten Steigungen zurück und setzt den Akkumulator zurück. Bei ungleichen Mikrobatchgrößen werden die Gewichtsverluste anhand der Probenanzahl berechnet, anstatt die obige Gleichgewichtsteilung zu verwenden.

Rufen Sie `scheduler.step()` einmal pro Parameteraktualisierung auf, nicht nach jedem Mikrobatch. Erstellen Sie es mit `StepLrSchedulerConfig::new(1e-3, 100).with_gamma(0.5).init()`: Die Lernrate beginnt bei `1e-3` und wird alle 100 Aufrufe mit `0.5` multipliziert. Die Initialisierung gibt `Result<StepLrScheduler, String>` zurück.

## Trainingsstatus speichern und wiederherstellen

`TrainingRecord` speichert das Modell, den Optimierer, den Lernratenplaner, die ausstehenden akkumulierten Gradienten und den Aufruferstatus zusammen. Diese Funktionen verwenden die oben genannten Typen wieder und erhalten den aktiven Trainingsstatus. Durch das Speichern wird kein neuer Optimierer, Planer oder Akkumulator erstellt.

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

Übergeben von `Path::new("checkpoints/step-100")` schreibt `checkpoints/step-100.bin`. `completed_updates` zählt abgeschlossene Parameteraktualisierungen und `pending_microbatches` zählt akkumulierte Mikrobatches im aktuellen Fenster. Rufen Sie nach der Wiederherstellung beide Zähler von `restored.state` ab.

Wiederherstellung mit derselben Modellstruktur, derselben Adam-Konfiguration und derselben Scheduler-Konfiguration, die beim Speichern verwendet wurden. Weiter mit `restored.accumulator`; Löschen Sie es nicht vorzeitig und spielen Sie die bereits angesammelten Mikrobatches nicht erneut ab. Versetzen Sie die Dateniterationsposition und den RNG-Status in den Aufruferstatus `U` und stellen Sie sie wieder her, bevor Sie den nächsten Stapel abrufen. `TrainingRecord` erstellt nicht automatisch einen Snapshot eines DataLoader.

## Wählen Sie einen anderen Optimierer

`ruda_optim` bietet außerdem `SgdConfig`, `AdamWConfig`, `AdaGradConfig`, `RmsPropConfig`, `AdanConfig`, `MuonConfig` und `LBFGSConfig`. Ändern Sie beim Wechseln des Optimierers sowohl die Konfiguration als auch den Statustyp. Informationen zur kollektiven Schulungsintegration finden Sie unter [ruCCL](libraries/ruccl.md).

API Referenz: [Optimierer](../../ruda-optim/src/optim/mod.rs), [Trainingsaufzeichnungen](../../ruda-optim/src/training.rs).
