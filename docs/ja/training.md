# トレーニングと状態の保存

[English](../en/training.md) | [简体中文](../zh/training.md) | **日本語** | [Deutsch](../de/training.md) | [Русский](../ru/training.md)

[ドキュメント](README.md) · [テンソル フレームワーク](tensor-framework.md) · [中文](../zh/training.md)

## トレーニング バックエンドを構成する

`Autodiff<Cuda<f32, i32>>` で CUDA テンソルの逆伝播グラフを記録し、`ruda-nn` で層を定義し、`ruda-optim` でパラメータを更新します。NVIDIA の環境構築は[はじめに](getting-started.md)を参照してください。

これらの依存関係をアプリケーションの `Cargo.toml` に追加します。この例では、アプリケーション ディレクトリを `RUDA` ソース ディレクトリの横に配置します。

```toml
[dependencies]
ruda-autodiff = { path = "../RUDA/ruda-autodiff", default-features = false, features = ["std"] }
ruda-model = { path = "../RUDA/ruda-model", default-features = false, features = ["std"] }
ruda-nn = { path = "../RUDA/ruda-nn", default-features = false, features = ["std"] }
ruda-optim = { path = "../RUDA/ruda-optim", default-features = false, features = ["std"] }
ruda-tensor-device = { path = "../RUDA/ruda-tensor-device", default-features = false, features = ["std", "cuda"] }
```

## 前方、後方、パラメータの更新

このトレーニング ステップでは、実際のバッチ `x` を取得し、`y` をターゲットにし、平均二乗誤差を計算し、線形レイヤーを更新します。 2 つの入力フィーチャーと 1 つの出力の場合、`x` の形状は `[batch, 2]` であり、`y` の形状は `[batch, 1]` です。どちらもバックエンド `B`、同じデバイス、および F32 データを使用します。

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

モデルとオプティマイザーを初期化し、バッチと一緒に `train_step` に渡します。

```rust
fn initialize(device: &CudaDevice) -> (Model, AdamOptimizer) {
    let model = LinearConfig::new(2, 1).init::<B>(device);
    let optimizer = AdamConfig::new().init();
    (model, optimizer)
}
```

`backward()` は損失テンソルを消費し、勾配を生成します。 `GradientsParams::from_grads` は、それらをモデル パラメーターに関連付けます。各 `optimizer.step` 呼び出しによって返される新しいモデルを保持し、ステップ間でオプティマイザーを保持して、Adam の運動量状態を保持します。

## 勾配の累積と学習率のスケジューリング

完全なバッチがデバイス メモリに収まらない場合は、パラメータを 1 回更新する前に、いくつかのマイクロバッチを前後に実行します。この関数では、すべてのマイクロバッチで等しいサンプル数が必要です。各平均損失をマイクロバッチの数で割ると、結合された平均損失の勾配が生成されます。

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

`accumulate` は平均化せずにグラデーションを追加します。 `grads()` は、累積された勾配を返し、アキュムレータをリセットします。マイクロバッチ サイズが等しくない場合は、上記の均等重量分割を使用する代わりに、サンプル数による重量損失が発生します。

マイクロバッチごとではなく、パラメータ更新ごとに 1 回 `scheduler.step()` を呼び出します。 `StepLrSchedulerConfig::new(1e-3, 100).with_gamma(0.5).init()` で作成します。学習率は `1e-3` から始まり、100 呼び出しごとに `0.5` で乗算されます。初期化により `Result<StepLrScheduler, String>` が返されます。

## トレーニング状態の保存と復元

`TrainingRecord` は、モデル、オプティマイザー、学習率スケジューラ、保留中の累積勾配、呼び出し元の状態をまとめて保存します。これらの関数は上記のタイプを再利用し、アクティブなトレーニング状態を受け取ります。保存しても、新しいオプティマイザー、スケジューラー、またはアキュムレーターは作成されません。

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

`Path::new("checkpoints/step-100")` を渡すと、`checkpoints/step-100.bin` が書き込まれます。 `completed_updates` は完了したパラメータ更新をカウントし、`pending_microbatches` は現在のウィンドウ内に蓄積されたマイクロバッチをカウントします。復元後、`restored.state` から両方のカウンタを取得します。

保存時に使用したのと同じモデル構造、Adam 構成、およびスケジューラー構成を使用して復元します。 `restored.accumulator` に進みます。早期にクリアしたり、すでに蓄積されているマイクロバッチを再実行したりしないでください。データ反復位置と RNG 状態を呼び出し側状態 `U` に置き、次のバッチをフェッチする前にそれらを復元します。 `TrainingRecord` は、DataLoader のスナップショットを自動的に作成しません。

## 別のオプティマイザーを選択してください

`ruda_optim` は、`SgdConfig`、`AdamWConfig`、`AdaGradConfig`、`RmsPropConfig`、`AdanConfig`、`MuonConfig`、および `LBFGSConfig` も提供します。オプティマイザーを切り替えるときは、構成と状態タイプの両方を変更します。集合的なトレーニングの統合については、[ruCCL](libraries/ruccl.md) を参照してください。

API 参照: [オプティマイザー](../../ruda-optim/src/optim/mod.rs)、[トレーニング レコード](../../ruda-optim/src/training.rs)。
