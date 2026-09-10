# ruBLAS ユーザーガイド

[English](../../en/libraries/rublas.md) | [简体中文](../../zh/libraries/rublas.md) | **日本語** | [Deutsch](../../de/libraries/rublas.md) | [Русский](../../ru/libraries/rublas.md)

[計算ライブラリ](README.md) · [ランタイム API](../runtime-api.md) · [中文](../../zh/libraries/rublas.md)

## 1. 概要と特徴

ruBLAS は、線形代数演算とデバイス テンソル インターフェイスを提供します。

| feature |モジュール|
| --- | --- |
|`tensor-vector`|`rublas::tensor_vector`|
|`tensor-matmul`|`rublas::tensor_matmul`|
|`tensor-matmul-autotune`|テンソル行列乗算オートチューニング|
|`tensor-int4`|`rublas::tensor_int4`|
|`tensor-grouped`|`rublas::tensor_grouped`|

Cargo パッケージは `rublas` です。必要な機能を選択し、汎用パスの不要なデフォルトを無効にします。 [Cargo.toml](../../../ruBLAS/Cargo.toml)を参照してください。

## 2. 行列の乗算、ベクトル、および INT4

### テンソル行列の乗算

`rublas::tensor_matmul::matmul` は lhs、rhs、省略可能な出力 out、MatmulStrategy、出力 DType を受け取り、デバイステンソルまたは MatmulSetupError を返します。既存の出力を out に渡すか、省略して新しく確保します。

戦略には Ruda、CmmaResidueFirst、Naive、および `tensor-matmul-autotune` が有効な場合の Autotune があります。この feature が無効なら既定は Ruda、有効なら Autotune です。量子化入力では Naive が最初の経路の失敗後に逆量子化して計算する場合がありますが、これはネイティブ INT4 計算ではありません。

`CmmaResidueFirst` は、実体化されたパディング行列を必要とせずに、部分的な K32 タイルを最初に処理する Tensor コア パスを明示的に選択します。 lhs および rhs には、一致する量子化されていない BF16 または F16 dtype と互換性のある行列次元が必要です。デバイスはパスのマトリックス命令をサポートする必要があります。この関数は F32 出力を返します。

```rust
use ruda_core::tensor::DType;
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use rublas::{
    kernel_ir::definition::MatmulSetupError,
    tensor_matmul::{MatmulStrategy, matmul},
};

fn residue_matmul<R: Runtime>(
    lhs: RudaTensor<R>,
    rhs: RudaTensor<R>,
) -> Result<RudaTensor<R>, MatmulSetupError> {
    matmul(lhs, rhs, None, MatmulStrategy::CmmaResidueFirst, DType::F32)
}
```

入力 `[M, K]` および `[K, N]` は、`[M, N]` を生成します。無効なセットアップまたは欠落しているデバイス機能がある場合は、Naive に切り替えるのではなく、`MatmulSetupError` を返します。

[行列乗算エントリ ポイント](../../../ruBLAS/src/tensor_matmul/base.rs) を参照してください。

### ベクトルの外積

`rublas::tensor_vector::cross(lhs, rhs, dim)` は選択した次元の長さが 3 であることを要求し、デバイステンソルを返します。最後以外の次元で計算する場合は、軸の置換と連続配置への変換を伴います。[cross.rs](../../../ruBLAS/src/tensor_vector/cross.rs) を参照してください。

### AWQ INT4

`rublas::tensor_int4::AwqGemm::new` は、パックされた重みを構築するために、qweight、qzeros、scales、オプションの bias、および group_size を受け取ります。 `forward` は、F16 入力を受け入れ、F16 出力または Int4Error を返します。

入力次元 K、出力次元 N、およびグループ サイズ G の場合:

|データ|Dタイプ|形状|
| --- | --- | --- |
|qweight| パック済み I32 |[K, N/8]|
|qzeros| パック済み I32 |[K/G, N/8]|
|scales|F16|[K/G, N]|
|bias (オプション)|F16|[N]|

K、N、および G はゼロ以外でなければなりません。 K は G で割り切れ、N は 8 で割り切れる必要があります。入力の最終次元は K で、出力では N に置き換えられます。すべてのオペランドはデバイスを共有する必要があります。パックされたビット順序は [AWQ カーネル](../../../ruBLAS/src/tensor_int4/kernel.rs) と一致する必要があります。任意の INT4 ファイルを直接 qweight として使用することはできません。

オブジェクトとそのチェックについては、[INT4 インターフェイス](../../../ruBLAS/src/tensor_int4/mod.rs) を参照してください。

## 3. グループ化された行列の乗算

`rublas::tensor_grouped::grouped_matmul_nt<R: Runtime>` は、入力、重み、および row_experts を `RudaTensor<R>` 値として受け取ります。 `Result<RudaTensor<R>, GroupedMatmulError>` を返します。

|引数|形状|要件|
| --- | --- | --- |
|入力|[M, K]|量子化されていない F32、F16、または BF16|
|の重み|[E, N, K]| 入力と同じ dtype とデバイス |
|row_experts|[M]|同じデバイス上の量子化されていない U32|
|出力|[M, N]|入力と同じ dtype|

行 m は、row_experts[m] でインデックス付けされた重み行列を選択し、入力行とその行列の各行の間のドット積を計算します。最後の 2 つの重み次元は、具体化された転置を必要とせずに、転置された形式で参加します。

## 4. グループ化された乗算セマンティクス

- K、N、および E は正でなければなりません。 Ｍはゼロであってもよい。
- 有効範囲外のエキスパートインデックスはパディングを示し、対応する出力行はすべてゼロになります。
- カーネルは FP32 に蓄積し、入力 dtype にキャストします。
- エントリ ポイントは、必要に応じて入力と重みを連続させるため、データがコピーされる可能性があります。
- 関連する要素数は、U32 インデックスに適合する必要があります。
- 無効な引数は `GroupedMatmulError` を返します。リードバックまたは同期中の非同期実行エラーを処理します。

ソース: [グループ化されたインターフェイス](../../../ruBLAS/src/tensor_grouped/mod.rs) および [カーネル](../../../ruBLAS/src/tensor_grouped/kernel.rs)。

## 5. 他のライブラリとの統合

[ruDNN MoE](rudnn.md) は、エキスパート投影にグループ化された乗算を使用します。量子化された重みは別の `tensor_int4` モジュールを使用します。通常の浮動小数点グループ乗算は、INT4 専門家による計算ではありません。

グループ化されたカーネルはスカラー累積を使用します。 dtype、形状、バックエンドの行列乗算戦略を測定します。
