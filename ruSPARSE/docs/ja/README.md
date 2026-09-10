# ruSPARSE

[English](../../README.md) | [简体中文](../zh/README.md) | **日本語** | [Deutsch](../de/README.md) | [Русский](../ru/README.md)

**英語** | [简体中文](../zh/README.md)

このリポジトリはソース ミラーです。 [RUDA monorepo](https://github.com/shuqi2077/RUDA) ルートから以下のコマンドを実行します。

Ruda のスパース 計算ライブラリ。共有テンソル、ランタイム、デバイス バックエンドを再利用します。

## 機能とエントリ ポイント

| feature |エントリ ポイントと依存関係|
| --- | --- |
|デフォルト (空)|スパース形式、検証、およびホスト変換。デバイス実装は含まれません|
|`tensor`|`CsrTensor<R>`、SpMV、SpMM、SpVV、SDDMM、SpGEAM、およびSpGEMM。呼び出し元が共有ランタイムを選択する|
|`cuda`|`tensor` と NVIDIA ドライバー (CUDA プログラムおよび例用)|
|`cuda-tests`|`cuda` テストの互換性エントリ ポイント|

## CUDA の例

```sh
cargo run -p ruSPARSE --features cuda --example csrmv
```

この例では、`alpha * A * x + beta * y` を計算し、結果を同期的に読み取り、`[7.0, 2.0, 18.5]` と照合します。

## 現在の計算契約

- 共有デバイス パスは、FP32 値と U32 インデックスを使用します。数値カーネルには 32 レーンのサブグループが必要です。
- `CsrTensor::from_csr` で検証済みの構造をアップロードします。`with_values` は値を更新し、`transpose` は現在のデバイス上の値を並べ替えます。演算は入力を上書きせず、新しい結果を返します。
- SpGEAM/SpGEMM はホスト側のシンボリック アルゴリズムを共有し、明示的なゼロ構造を保持します。数値ステージはデバイス上で実行されます。構造インデックスはホストのコピーを保持します。値はホスト側の計算のために読み戻されません。

## ruSPARSE ユーザーガイド

[計算ライブラリ](../../../docs/ja/libraries/README.md) · [中文](../zh/README.md)

### 1. 概要

ruSPARSE はスパース計算を提供します。

### 2. データ記述子

|タイプ|目的|
| --- | --- |
|`CsrMatrix`、`CsrMatrixOwned`|借用した CSR ビューと所有ストレージ|
|`CooMatrix`、`CooMatrixOwned`|COO 表現|
|`CscMatrix`、`CscMatrixOwned`|CSC 表現|
|`BsrMatrix`、`EllMatrix`|ブロック スパース表現と ELL 表現|
|`DenseMatrix`、`DenseMatrixOwned`|密行列記述子|
|`SparseVector`|スパースベクトル|
|`IndexBase`| インデックスの基数は Zero または One |
|`DenseOrder`| RowMajor または ColumnMajor |
|`Operation`| None、Transpose、ConjugateTranspose |

インデックス ベース、レイアウト、および操作タイプは明示的な規約であり、配列の長さだけから推測することはできません。 Owned は、所有されているデータと借用されたビューを区別します。汎用 GPU ストレージは自動的に割り当てられません。

### 3. 操作方法

演算は、行列ベクトル乗算、行列乗算、スパース行列乗算/加算、スパース ベクトル ドット積、SDDMM、およびフォーマット変換に分類されます。実行レイヤーには、`SparsePlan`、`SparseError`、および操作固有の結果タイプが含まれます。

CSR 行列とベクトルの乗算では、alpha、op(A)、x、beta、y 契約を使用します。呼び出しを移植するときは、ライブラリ接頭辞を変更するだけでなく、インデックス ベース、転置モード、およびデータ レイアウトを一致させます。

### 4. CUDA CSR 行列ベクトル乗算

Cargo パッケージは `ruSPARSE` です。 Rust インポート名は `rusparse` です。機能 `tensor` は `rusparse::tensor` を有効にします。 `cuda` は、デバイス テンソルと CUDA 依存関係の両方を有効にします。ソースルートから実行します。

```powershell
cargo run --locked -p ruSPARSE --features cuda --example csrmv
```

この例では、`0.5 × A × x + 2 × y` を計算します。

```rust
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use rusparse::{
    CsrMatrix, IndexBase, Operation,
    tensor::{CsrTensor, csrmv},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let host = CsrMatrix::new(
        3, 3,
        &[0, 2, 3, 4],
        &[0, 1, 1, 2],
        &[2., 1., -1., 5.],
        IndexBase::Zero,
    )?;
    let matrix = CsrTensor::<CudaRuntime>::from_csr(host, Operation::None, &device)?;
    let x = from_data(TensorData::new(vec![3f32, 4., 5.], [3]), &device);
    let y = from_data(TensorData::new(vec![1f32, 2., 3.], [3]), &device);
    let output = into_data_sync(csrmv(&matrix, 0.5, x, 2., y)?).to_vec::<f32>()?;
    assert_eq!(output, [7., 2., 18.5]);
    println!("alpha * A * x + beta * y = {output:?}");
    Ok(())
}
```

`CsrMatrix::new` の引数は、行、列、行オフセット、列インデックス、ゼロ以外の値、およびインデックス ベースです。最初の行には 2 つのエントリが格納されます。 2 番目と 3 番目の店舗に 1 つずつあります。結果は`[7.0, 2.0, 18.5]`です。

`CsrTensor::from_csr` は構造と値をアップロードします。行列を転置するには、ここで `Operation::Transpose` を指定します。 `csrmv(&matrix, alpha, x, beta, y)` の場合、x の長さは列数と等しく、y の長さは行数と等しくなります。密な入力は、行列のデバイス上で量子化されていない F32 テンソルである必要があります。この呼び出しは新しいデバイス テンソルを返します。マトリックスを再利用するには、CSR 構造を再度アップロードする必要はありません。

疎/密行列の乗算には、`csrmm(&matrix, operation_b, alpha, b, beta, c, output_order)` を使用します。ここで c はオプションで、`DenseOrder` は出力レイアウトを指定します。ゼロ以外の値のみを変更するには、長さ nnz の F32 デバイス テンソルを指定して `matrix.with_values(values)` を呼び出します。構造は保存されます。

### 5. テンソル フレームワークの CSR

`ruda_tensor::api::CsrTensor<B>` は、`SparseOps` を実装するバックエンドをターゲットとします。上記の`rusparse::tensor::CsrTensor<R>`とは別タイプです。 `CsrTensor::<B>::from_data(&data, &device)` で作成します。データのタイプはバックエンドの `B::CsrData` です。

|メソッド|使用法|
| --- | --- |
|`matmul(rhs)`、`transpose_matmul(rhs)`|スパース行列またはその転置行列を 2 次元の密テンソルで乗算します。|
|`sparse_matmul(&rhs)`|スパース行列を乗算し、出力パターンを構築します|
|`add(&rhs)`、`add_scaled(&rhs, alpha, beta)`|スパース行列を追加します (オプションで係数を使用)|
|`gather(dense)`|CSR 位置の密な値を 1 次元テンソルに収集します|
|`scatter_add()`|保存された値を密な位置に追加します|
|`mul_dense(rhs)`|CSR 位置の密な値を乗算します。|
|`sampled_matmul(lhs, rhs)`|現在の CSR 位置でのみ密行列積を評価します|
|`sampled_sparse_matmul(&lhs, &rhs)`|現在の CSR 位置でのみ疎行列積を評価します|
|`transpose()`、`to_dense()`|転置または密テンソルへの変換|
|`with_values(values)`|構造を保持したまま 1 次元値テンソルを置き換えます|
|`to_data().await`|非同期リードバック `B::CsrData`|

これらの計算メソッドは `Result` を返します。フレームワークのスパース操作で使用される値テンソルは量子化されていない必要があります。

API 参照: [デバイス CSR](../../src/tensor/mod.rs)、[フレームワーク CSR](../../../ruda-tensor/src/api/sparse.rs)。
