# ruTENSOR ユーザーガイド

[English](../../en/libraries/rutensor.md) | [简体中文](../../zh/libraries/rutensor.md) | **日本語** | [Deutsch](../../de/libraries/rutensor.md) | [Русский](../../ru/libraries/rutensor.md)

[計算ライブラリ](README.md) · [Tensor フレームワーク](../tensor-framework.md) · [中文](../../zh/libraries/rutensor.md)

ruTENSOR は、名前付き軸テンソル縮約、リダクション、物理的置換、および要素ごとの操作を提供します。入力は `RudaTensor<R>` を使用します。アプリケーションはデバイスのランタイムを選択します。このライブラリは、上位レベルの `ruda-tensor` フレームワークとは異なります。

## 1. 依存関係を構成する

Cargo パッケージは `ruTENSOR` です。 Rust インポート名は `rutensor` です。デフォルトでは、ドライバーを選択せずに、`std` とデバイス テンソル計算が有効になります。次の構成では、アプリケーション ディレクトリを `RUDA` ソース ディレクトリの横に配置します。

```toml
[dependencies]
rutensor = { package = "ruTENSOR", path = "../RUDA/ruTENSOR" }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

## 2. 縮約、帰約、置換

以下をアプリケーションの `src/main.rs` として保存します。

```rust
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use rutensor::{einsum, permute, reduce, ReductionOp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let a = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]), &device,
    );
    let b = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 0., 0., 1., 1., 1.], [3, 2]), &device,
    );
    let product = einsum("ik,kj->ij", &[&a, &b])?;
    let row_sums = reduce(&a, &[0, 1], &[0], ReductionOp::Sum)?;
    let transposed = permute(&a, &[1, 0])?;

    println!("product: {:?}", into_data_sync(product).to_vec::<f32>()?);
    println!("row sums: {:?}", into_data_sync(row_sums).to_vec::<f32>()?);
    println!("transpose: {:?}", into_data_sync(transposed).to_vec::<f32>()?);
    Ok(())
}
```

`einsum("ik,kj->ij", ...)` は k を合計し、形状 `[2, 2]` を返します。 `reduce` はモード 0 を保持し、モード 1 を縮小し、形状 `[2]` を返します。 `permute` は、入力ストレージを共有するビューではなく、新しく割り当てられた `[3, 2]` テンソルを返します。

## 3. Einsum 式

`einsum(expression, inputs)` は 1 つ以上の入力を受け入れます。大文字と小文字が区別される文字は軸を識別します。矢印の右側は、出力軸を選択して順序付けします。

|式|オペレーション|
| --- | --- |
|`ik,kj->ij`|行列乗算|
|`...ik,...kj->...ij`|ブロードキャストバッチ軸による行列乗算|
|`abc,cde->abde`| 多次元テンソル縮約 |
|`ij,jk,kl->il`| 3 入力の縮約 |
|`i,j->ij`| 外積（テンソル積） |
|`ii->i`| 対角成分 |
|`ii->`|トレース、ランク 0 のスカラーを返す|
|`ijk->ki`| j を帰約し、残りの軸を並べ替える |
|`...i->i`| 省略記号が表すすべての軸を帰約する |

- 入力間の一致モードは、等しい範囲または 1 の範囲を持つ必要があります。省略記号軸は右揃えでブロードキャストされます。
- 入力内で繰り返されるラベルは対角線を選択します。これらの軸の範囲は正確に等しい必要があります。
- 出力ラベルは一意であり、入力内に存在する必要があります。
- `->` を使用しない場合、出力軸は省略記号軸で始まり、その後に 1 回だけ出現するアルファベット順にソートされたラベルが続きます。
- スカラー入力は空のラベル セグメントを使用します。たとえば、`,ij->ij` は、行列への最初のスカラー入力を乗算します。
- 入力は連続していなくてもかまいません。計算ではテンソル値がホストに読み戻されません。

固定式、シェイプ、ストライド、および dtype の場合は、`EinsumPlan::new(expression, descriptors)` を構築し、`execute(inputs)` を再利用します。出力と算術精度を指定するには、`EinsumPlan::with_options` または `einsum_with_options` を使用します。

## 4. 記述子と実行プラン

`Mode` は `i32` ラベルです。同じラベルは、物理的な位置に関係なく、テンソル全体で同じ論理軸を識別します。 `TensorDescriptor` は、エクステント、要素ストライド、およびストレージ dtype を格納します。 `OperandDescriptor` は、ラベルと入力単項変換を追加します。

これらの関数は、`D = alpha * A @ B + beta * C` のプランを作成および実行します。

```rust
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use rutensor::{
    ComputeType, DType, OperandDescriptor, OperationDescriptor,
    Plan, Result, TensorDescriptor,
};

fn make_plan<R: Runtime>(
    a: &RudaTensor<R>, b: &RudaTensor<R>, c: &RudaTensor<R>,
    m: usize, n: usize,
) -> Result<Plan> {
    let operation = OperationDescriptor::contraction(
        OperandDescriptor::from_tensor(a, &[0, 2])?,
        OperandDescriptor::from_tensor(b, &[2, 1])?,
        Some(OperandDescriptor::from_tensor(c, &[0, 1])?),
        TensorDescriptor::contiguous(&[m, n], DType::F32)?,
        &[0, 1],
        ComputeType::F32,
    )?;
    Plan::new(operation)
}

fn execute<R: Runtime>(
    plan: &Plan, a: &RudaTensor<R>, b: &RudaTensor<R>, c: &RudaTensor<R>,
    alpha: f64, beta: f64,
) -> Result<RudaTensor<R>> {
    plan.execute(&[a, b, c], &[alpha, beta])
}
```

プランは入力バッファを保持しません。後続の実行では、形状、ストライド、dtype が一致する異なるテンソルを使用できます。すべての入力は同じデバイス上に存在する必要があります。

|コンストラクター|実行入力順序|スカラー順序|
| --- | --- | --- |
|`contraction`| A、B（C は省略可能） | alpha（C がある場合は beta も） |
|`sum_product`| 積のすべての入力（C は省略可能） | alpha（C がある場合は beta も） |
|`reduction`| A（C は省略可能） | alpha（C がある場合は beta も） |
|`permutation`| A | alpha |
|`elementwise_binary`| A, B | alpha, beta |
|`elementwise_trinary`| A, B, C | alpha, beta, gamma |

`Plan::execute` は出力を割り当てます。 `Plan::execute_into` は、形状、ストライド、および dtype が出力記述子と一致する出力テンソルを受け入れて返します。そのバッファは、入力や他のビューと共有せずに、排他的に所有する必要があります。

パディングまたは並べ替えられた出力レイアウトには `TensorDescriptor::new(extents, strides, dtype)` を使用します。出力軸は重なってはいけません。明示的な操作記述子は、出力形状を通じて追加のブロードキャスト軸を導入できます。

## 5. リダクションと要素ごとの演算

`reduce(input, input_modes, output_modes, operation)` は、出力に含まれないラベルを減らします。縮小された軸が削除されます。

|ReductionOp|オペレーション| 空の帰約 |
| --- | --- | --- |
|`Sum`|合計|0|
|`Product`| 積 |1|
|`Min`|最小値|正の無限大|
|`Max`|最大値|負の無限大|

`elementwise_binary` および `elementwise_trinary` は、`BinaryOp::{Add, Mul, Min, Max}` を使用して、名前付き軸を位置合わせしてブロードキャストします。三項演算は `(alpha * op(A) op_ab beta * op(B)) op_abc gamma * op(C)` を評価します。

`Identity`、`Negate`、`Abs`、`Sqrt`、`Exp`、`Log`、`Sin`、`Cos`、`Tanh`、を選択します。 `Relu`、`Reciprocal`、または `Conjugate` ～ `OperandDescriptor::with_unary`。 `Log` は自然対数です。実数値の場合、`Conjugate` は `Identity` と同じです。最小値と最大値は NaNs を伝播します。

## 6. 精度、ストレージ、エラー

- ストレージ タイプは、F16、BF16、F32、および F64 です。量子化、整数、および複素数のストレージは受け入れられません。
- `ComputeType::F32` または `F64` は、入力変換、単項変換、積、リダクション、およびスカラー演算を制御します。 F64 入力には、F64 の計算が必要です。
- デフォルトでは、入力が F64 の場合は F64 算術を使用し、それ以外の場合は F32 を使用します。入力ストレージ タイプが同じであれば、その出力タイプが保持されます。混合入力では、いずれかの入力が F64 の場合は F64 が生成され、それ以外の場合は F32 が生成されます。
- 高精度のストレージのために明示的な出力 dtype を要求します。ストレージの精度を高めても、入力ですでに失われた精度を回復することはできません。
- 出力変換は最後の書き込み時に行われます。コンピューティング タイプとは異なる dtype の入力には、デバイス変換バッファーが必要です。一致する入力は元のレイアウトで読み取られます。
- 削減では、並列ワークグループ ツリーの結合が使用されます。浮動小数点の加算と乗算の順序はシリアル評価とは異なる場合があります。
- 空の出力は計算を送信しません。ランク 0 の出力には 1 つのスカラーが含まれます。形状、ストライド、アドレスの計算にはプラットフォーム `usize` を使用します。ディスパッチはバックエンドのリソース制限にも従います。
- 返された `Result` 値は、式、記述子、形状、デバイス、およびディスパッチ範囲のエラーを報告します。デバイスの送信は、ランタイムのエラー処理に従います。非同期実行エラーは、同期またはリードバック時に発生します。
- 一般的な縮約は帰約対象の座標を直接横断します。これらは、複数入力の縮約順序を検索したり、自動的に Tensor Core 行列乗算に引き下げたりすることはありません。ストレージの変換には追加のデバイス メモリが必要です。

ruTENSOR は、NVIDIA cuTENSOR C ABI ではなく、Ruda Rust API を公開します。デバイスは、選択したストレージとコンピューティングの種類をサポートする必要があります。
