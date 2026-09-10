# ruPRIM ユーザーガイド

[English](../../en/libraries/ruprim.md) | [简体中文](../../zh/libraries/ruprim.md) | **日本語** | [Deutsch](../../de/libraries/ruprim.md) | [Русский](../../ru/libraries/ruprim.md)

[計算ライブラリ](README.md) · [Tensor フレームワーク](../tensor-framework.md) · [中文](../../zh/libraries/ruprim.md)

ruPRIM は、デバイス テンソル削減、累積スキャン、要素ごとの操作、およびインデックス付けを提供します。このページでは `RudaTensor<R>` を使用します。R はデバイス ランタイムです。

## 1. 依存関係を構成する

Cargo パッケージは `ruPRIM` です。 Rust インポート名は `ruprim` です。必要な操作の機能を有効にします。

| feature |インターフェイス|
| --- | --- |
|`tensor-reduce`|`ruprim::reduce::tensor`: 全テンソルと軸の縮小|
|`tensor-reduce-autotune`| 帰約の自動チューニング。tensor-reduce も有効化する |
|`tensor-scan`|`ruprim::scan`: 累積和、積、最小値、最大値|
|`elementwise`|`ruprim::elementwise`: 要素ごとの計算|
|`indexing`| `ruprim::indexing`: 選択、スライス、gather、scatter。elementwise も有効化する |

この構成では、アプリケーション ディレクトリが `RUDA` ソース ディレクトリの横に配置されます。 NVIDIA のセットアップについては、[はじめに](../getting-started.md) を参照してください。

```toml
[dependencies]
ruprim = { package = "ruPRIM", path = "../RUDA/ruPRIM", default-features = false, features = ["std", "tensor-reduce", "tensor-scan", "indexing"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

## 2. 合計、行削減、およびスキャン

この完全な `src/main.rs` は、F32 行列 `[[1, 2, 3], [4, 5, 6]]` を使用して、その合計、行平均、行 argmax インデックス、および行プレフィックス合計を計算します。アプリケーション ディレクトリから `cargo run` を実行します。

```rust
use ruda_core::tensor::{DType, TensorMetadata, data::TensorData};
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use ruprim::reduce::{
    components::instructions::ReduceOperationConfig,
    tensor::{KernelReduceStrategy, SumStrategy, reduce_dim, sum},
};
use ruprim::scan::cumsum;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let input = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]),
        &device,
    );
    let total = sum(
        input.clone(),
        SumStrategy::Chained(KernelReduceStrategy::Unspecified),
    )?;
    assert_eq!(total.shape().as_slice(), &[1]);
    assert_eq!(into_data_sync(total).to_vec::<f32>()?, [21.0]);

    let means = reduce_dim(
        input.clone(), None, 1,
        KernelReduceStrategy::Unspecified, ReduceOperationConfig::Mean,
    )?;
    assert_eq!(means.shape().as_slice(), &[2, 1]);
    assert_eq!(into_data_sync(means).to_vec::<f32>()?, [2.0, 5.0]);

    let indices = reduce_dim(
        input.clone(), Some(DType::I32), 1,
        KernelReduceStrategy::Unspecified, ReduceOperationConfig::ArgMax,
    )?;
    assert_eq!(into_data_sync(indices).to_vec::<i32>()?, [2, 2]);

    let prefix = cumsum(input, 1);
    assert_eq!(into_data_sync(prefix).to_vec::<f32>()?, [1., 3., 6., 4., 9., 15.]);
    Ok(())
}
```

`sum` はすべての要素を削減し、形状 `[1]` を返します。 `reduce_dim` は選択された軸のみを処理し、通常はランクを保持し、その軸の長さを 1 に設定します。したがって、行平均の形状は `[2]` ではなく、`[2, 1]` になります。

## 3. 削減活動と戦略

テンソル インターフェイスの完全なパラメーター順序:

|関数|目的|
| --- | --- |
|`sum(tensor, strategy)`|全テンソル和|
|`sum_fallback(tensor, strategy)`|全テンソルの合計。必要なアトミック追加が利用できない場合、OneShot を Chained に切り替えます|
|`reduce(tensor, output_dtype, strategy, config)`|すべての軸を順番に縮小し、形状を返します [1]|
|`reduce_dim(tensor, output_dtype, dim, strategy, config)`|選択した軸を縮小します|

`config` と `ReduceOperationConfig` を選択します。

|オペレーション|`reduce_dim`出力|
| --- | --- |
|`Sum`、`Prod`、`Mean`|合計、積、平均。選択された軸の長さ 1|
|`Min`、`Max`、`MaxAbs`|絶対値の最小値、最大値、最大値。選択された軸の長さ 1|
|`ArgMin`、`ArgMax`|選択した軸内のゼロベースのインデックス。軸長1|
|`TopK(k)`|選択した軸に沿った上位 k 個の値。軸長さ k|
|`ArgTopK(k)`|軸内の対応するインデックス。軸長さ k|

インデックス削減には、`Some(DType::I32)` などの明示的な整数出力 dtype が必要です。値を削減するには、`None` を渡します。出力は入力 dtype を保持します。 `output_dtype` を一般的なキャスト パラメータとして使用しないでください。 F16/BF16 Sum、Prod、Mean は、入力 dtype を書き込む前に、このパスで FP32 累積を使用します。軸は有効である必要があり、TopK k は `1..=axis length` にある必要があります。 TopK には、最終的な形状を `[1]` にリセットする全テンソル `reduce` ではなく、`reduce_dim` を使用します。

次のように `SumStrategy` を選択します。

- `OneShot(ruda_count)`: 正のワークグループ数を明示的に設定し、入力 dtype のアトミック加算を必要とします。
- `Chained(KernelReduceStrategy::Unspecified)`: 全テンソル和のグローバル アトミック加算を必要とせずに、段階的なリダクションを使用します。例で使用されています。
- `Autotune`: `tensor-reduce-autotune` で使用できます。この機能がない場合、デフォルトは `OneShot(4)` です。この場合のデフォルトは `Autotune` です。

`KernelReduceStrategy` は、`Unspecified`、`Specific(ReduceStrategy)`、および機能ゲート型 `Autotune` を提供します。低レベルの戦略を修正するには、Specific を使用します。自動チューニングを行わない場合、デフォルトは「未指定」です。

帰約インターフェースは `Result<RudaTensor<R>, ReduceError>` を返します。範囲外の軸は `InvalidAxis`、OneShot に必要なアトミック加算の欠如は `MissingAtomicAdd` を返します。`sum_fallback` が置き換えるのは未対応の OneShot アトミック加算の場合だけで、CPU 実行には切り替えません。

## 4. 累積スキャン

4 つの関数はすべて `(tensor, dim)` を受け取ります。

|関数|1 つの入力行の結果 [3、1、2]|
| --- | --- |
|`cumsum`|[3、4、6]|
|`cumprod`|[3、3、6]|
|`cummin`|[3、1、1]|
|`cummax`|[3、3、3]|

これらは包括的なプレフィックス操作です。出力形状と dtype は入力と一致します。バッチ行は独立して処理されます。 dim は 0 から始まり、入力ランク内になければなりません。スキャンは、`Result` ではなく、テンソルを直接返します。

現在のアルゴリズムは、各出力位置に対応するプレフィックスを読み取り、長さ n の軸に対して合計 O(n²) 回の読み取りを行います。入力要素数だけから作業を見積もるのではなく、長いスキャン軸でのこのコストを考慮してください。

## 5. 選択、収集、スライス、および整数累乗

この関数を同じファイルに追加し、メインが戻る前に `indexing_example(&device)?;` を呼び出します。以前のインポートを再利用します。

```rust
fn indexing_example(device: &CudaDevice) -> Result<(), Box<dyn std::error::Error>> {
    use ruda_core::tensor::element::Scalar;
    use ruprim::{
        elementwise::binary::integer_power,
        indexing::{gather, select, slice},
    };

    let input = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]),
        device,
    );
    let columns = from_data::<CudaRuntime>(
        TensorData::new(vec![2i32, 0], [2]), device,
    );
    let selected = select(input.clone(), 1, columns);
    assert_eq!(into_data_sync(selected).to_vec::<f32>()?, [3., 1., 6., 4.]);

    let positions = from_data::<CudaRuntime>(
        TensorData::new(vec![2i32, 0, 1, 1], [2, 2]), device,
    );
    let gathered = gather(1, input.clone(), positions);
    assert_eq!(into_data_sync(gathered).to_vec::<f32>()?, [3., 1., 5., 5.]);

    let sliced = slice(input.clone(), &[0..2, 1..3]);
    assert_eq!(into_data_sync(sliced).to_vec::<f32>()?, [2., 3., 5., 6.]);

    let squared = integer_power::scalar(input, Scalar::Int(2));
    assert_eq!(into_data_sync(squared).to_vec::<f32>()?, [1., 4., 9., 16., 25., 36.]);
    Ok(())
}
```

|関数|インデックスと出力|
| --- | --- |
|`select(tensor, dim, indices)`|1 次元の整数インデックス。選択した軸の長さをインデックス数に置き換えて、他の次元に同じ位置を適用します|
|`gather(dim, tensor, indices)`|dim は最初の引数です。出力位置ごとに入力値を選択し、出力形状はインデックス形状と同じになります。|
|`slice(tensor, ranges)`|軸ごとの半開 `start..end` 範囲。この例では、すべての行から列 1 と列 2 を選択します。|
|`slice_assign(tensor, slices, value)`|は `ruda_core::tensor::Slice` を使用します。選択した領域に値を書き込み、更新されたテンソルを返します|

インデックスとデータはデバイスを共有する必要があります。インデックスは有効なゼロベースの整数である必要があります。収集の場合は、同じランクと一致する選択されていないディメンションを持つインデックス テンソルを使用します。サンプル行ごとに異なる列を選択できます。スライス範囲は軸内にある必要があり、割り当て値の形状は選択した領域と一致する必要があります。

整数累乗は `ruprim::elementwise::binary::integer_power` にあります。

- `scalar(input, Scalar::Int(exponent))`: すべての要素に対して 1 つの整数指数を使用します。
- `tensor(base, exponents)`: ブロードキャスト互換の形状を持つ浮動小数点基数と整数指数テンソルを使用します。出力はベース dtype を維持します。

負の指数には、指数を浮動小数点に変換するのではなく、`Scalar::Int(-2)` を使用します。インデックス作成および要素ごとのインターフェイスは、デバイス テンソルを直接返します。他の操作の入力を保持するには、例のようにクローンされたハンドルを渡し、各操作によって返されたテンソルを保持します。

API 参照: [縮小](../../../ruPRIM/src/reduce/tensor/base.rs)、[スキャン](../../../ruPRIM/src/scan/tensor.rs)、[インデックス作成](../../../ruPRIM/src/indexing/mod.rs)、[整数乗](../../../ruPRIM/src/elementwise/binary/integer_power.rs)。
