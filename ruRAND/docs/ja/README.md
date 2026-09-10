# ruRAND

[English](../../README.md) | [简体中文](../zh/README.md) | **日本語** | [Deutsch](../de/README.md) | [Русский](../ru/README.md)

**英語** | [简体中文](../zh/README.md)

Ruda の乱数生成。

- Cargo パッケージ: `ruRAND`
- Rust クレート: `rurand`

## feature

| feature |操作|
| --- | --- |
|`tensor`|デバイス テンソルの一様分布、正規分布、およびベルヌーイ分布|

`rurand::tensor::{random_uniform, random_normal, random_bernoulli}` を使用してテンソルを割り当てるか、クレート ルートの関数を割り当てて既存のデバイス ストレージを埋めます。 `rurand::seed` は共有ランダム シードを設定します。

## クイック スタート

RUDA ワークスペースからビルドします。

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruRAND --no-default-features --features std,tensor
```

## ドキュメント

- [ユーザーガイド](../../../docs/ja/libraries/rurand.md)
- [環境設定](../../../docs/ja/getting-started.md)
- [Cargo 機能](../../Cargo.toml) · [モジュール エクスポート](../../src/lib.rs)

## ruRAND ユーザーガイド

[計算ライブラリ](../../../docs/ja/libraries/README.md) · [ランタイム API](../../../docs/ja/runtime-api.md) · [中文](../zh/README.md)

ruRAND は、一様分布、正規分布、およびベルヌーイ分布を使用してデバイス テンソルを生成します。 `rurand::tensor` を使用して、新しいテンソルを割り当てるか、クレート ルートにある同じ名前の関数を割り当てて、既存のデバイス ストレージを埋めます。

### 1. 依存関係を構成する

Cargo パッケージは `ruRAND` です。 Rust インポート名は `rurand` です。機能 `tensor` により、デバイス テンソル インターフェイスが有効になります。この構成では、アプリケーション ディレクトリが `RUDA` ソース ディレクトリの横に配置されます。 NVIDIA のセットアップについては、[はじめに](../../../docs/ja/getting-started.md) を参照してください。

```toml
[dependencies]
rurand = { package = "ruRAND", path = "../RUDA/ruRAND", default-features = false, features = ["std", "tensor"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

### 2. ランダム テンソルを生成する

この完全な `src/main.rs` は、形状 `[2, 3]` の F32 テンソルを生成し、シードをリセットすることによって呼び出しを再生します。アプリケーション ディレクトリから `cargo run` を実行します。

```rust
use ruda_core::tensor::DType;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::readback::into_data_sync;
use rurand::tensor::{
    random_bernoulli, random_like_uniform, random_normal, random_uniform,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    rurand::seed(42);
    let uniform = random_uniform::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let first = into_data_sync(uniform.clone()).to_vec::<f32>()?;

    rurand::seed(42);
    let repeated = random_uniform::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let repeated = into_data_sync(repeated).to_vec::<f32>()?;
    assert_eq!(first, repeated);
    assert!(first.iter().all(|&x| (0.0..1.0).contains(&x)));

    let normal = random_normal::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let mask = random_bernoulli::<CudaRuntime>([2, 3].into(), &device, 0.25, DType::F32);
    let noise = random_like_uniform(&uniform, -0.1, 0.1, DType::F32);

    let mask = into_data_sync(mask).to_vec::<f32>()?;
    assert!(mask.iter().all(|&x| x == 0.0 || x == 1.0));
    println!("uniform={first:?}");
    println!("normal={:?}", into_data_sync(normal).to_vec::<f32>()?);
    println!("mask={mask:?}");
    println!("noise={:?}", into_data_sync(noise).to_vec::<f32>()?);
    Ok(())
}
```

`random_like_uniform` は、リファレンスから形状とデバイスのみを取得します。参照値を変更またはコピーせずに、新しいテンソルを割り当てます。最後の引数は、出力 dtype を明示的に選択します。

`random_bernoulli` は、自動的に作成されたブール マスクではなく、選択した dtype の 0 と 1 を返します。この例では、F32 値 0.0 および 1.0 を返します。

### 3. パラメータと分布

最初の 3 つのテンソル関数は、`shape: Shape`、`device: &R::Device`、および最後の `dtype: DType` パラメーターを共有します。返された `RudaTensor<R>` には、指定された形状、dtype、およびデバイスが含まれます。

|関数とパラメータの順序|分布パラメータ|
| --- | --- |
|`random_uniform(shape, device, lower_bound, upper_bound, dtype)`|f32 下限と上限|
|`random_normal(shape, device, mean, std, dtype)`|f32 平均値と標準偏差。標準偏差は分散ではありません|
|`random_bernoulli(shape, device, probability, dtype)`|f32 1 を返す確率|
|`random_like_uniform(&reference, lower_bound, upper_bound, dtype)`|参照テンソル、f32 境界、出力 dtype|

分布と一致する有限パラメータを指定します: 上限より下の下限、非負の正規標準偏差、および `[0, 1]` のベルヌーイ確率。これらのインターフェイスは、`Result` を通じてこれらの範囲を検証しません。エラー戻りを要求するために無効なパラメータを使用しないでください。

均一生成では、最初に F32 `u ∈ [0, 1)` が生成され、`lower_bound + (upper_bound - lower_bound) × u` が計算されてから、出力 dtype にキャストされます。 F32 `[0, 1)` の場合は 1 を除きます。他の間隔にスケーリングするか、より低い精度に変換すると、上限に近い値がその境界に丸められる可能性があります。

通常の生成では、出力 dtype にキャストする前に、FP32 ボックス ミュラー変換を使用します。 F64 ストレージを選択しても、内部ランダム浮動小数点精度は向上しません。ベルヌーイは `u < probability` を比較するため、確率 0 はすべて 0 を生成し、確率 1 はすべて 1 を生成します。

### 4. シードと呼び出し順序

`rurand::seed(seed: u64)` は共有ホストのランダム状態をリセットします。世代呼び出しごとに、その状態から新しいシードが抽出され、それが進められます。

- シードを 1 回設定してシーケンスを初期化し、連続するバッチを生成します。
- シーケンスを再生するには、同じシードをリセットし、呼び出し順序、形状、dtype、およびデバイス構成を保存します。
- ランダムな入力を繰り返すことが意図されていない限り、各トレーニング バッチの前に同じシードをリセットしないでください。
- 同時スレッドはこの状態を共有します。インターリーブは、各呼び出しがどのシードを受け取るかに影響します。これは、スレッドまたはデバイスごとに個別のジェネレーター オブジェクトではありません。

上記の再生比較では、1 つのプロセス、1 つのデバイス、および一致する引数を使用します。シードの一致だけでは、別のランダム ライブラリまたはバックエンドとの要素ごとの同等性は意味されません。

### 5. 既存のバッファを埋める

クレートルート起動関数を使用して、出力割り当てを再利用します。この関数は、すでに割り当てられている F32 デバイス テンソルを取得し、ランダムな値をそのストレージに書き込み、それを返します。

```rust
use ruda_kernel::{
    dsl::{Runtime, prelude::LaunchError},
    tensor::RudaTensor,
};

fn fill_uniform<R: Runtime>(
    output: RudaTensor<R>,
    lower: f32,
    upper: f32,
) -> Result<RudaTensor<R>, LaunchError> {
    rurand::random_uniform(
        &output.client,
        lower,
        upper,
        output.clone().binding(),
        output.dtype.into(),
    )?;
    Ok(output)
}
```

この関数はテンソル アロケータを呼び出しません。そのストレージを共有する他のハンドルも書き込みを監視します。通常値とベルヌーイ値に対応する関数は、`rurand::random_normal(&client, mean, std, binding, dtype)` と `rurand::random_bernoulli(&client, probability, binding, dtype)` です。

Tensor レベルの関数は、起動時にエラーが返された場合にパニックを起こします。上記の基になるエントリ ポイントは `Result<(), LaunchError>` を返します。送信が成功しても、GPU の実行が完了したわけではありません。テンソルを読み取るか、クライアントの同期を待ちます。 `into_data_sync` はリードバック失敗時にパニックを起こします。非同期 `into_data(tensor).await` を使用してリードバック エラーを伝播します。

API リファレンス: [Tensor インターフェイス](../../src/tensor/mod.rs)、[シード状態](../../src/state.rs)、[ディストリビューション](../../src/distributions/mod.rs)。
