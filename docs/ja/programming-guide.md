# Ruda Programming Guide

[English](../en/programming-guide.md) | [简体中文](../zh/programming-guide.md) | **日本語** | [Deutsch](../de/programming-guide.md) | [Русский](../ru/programming-guide.md)

[ドキュメント](README.md) · [ランタイム API](runtime-api.md) · [計算ライブラリ](libraries/README.md) · [中文](../zh/programming-guide.md)

## 1. Host and device

ホスト Rust コードは、デバイスを選択し、入力を準備し、引数を構築し、結果を読み取ります。 Device kernels describe parallel computation.フロントエンドは、バックエンドのコンパイルと実行のためにそれらを IR に展開します。

`ruda-kernel::dsl` は汎用カーネル フロントエンドです。カーネルは、フロントエンドの型、マクロ、および操作で Rust 構文を使用します。任意の Rust プログラムと標準ライブラリ コードを単純に GPU にコンパイルすることはできません。

テンソル フレームワークは、演算ライブラリを計算するために操作をディスパッチします。アプリケーションは行列乗算を使用するためにスレッドレベルのカーネルを作成する必要はありません。

## 2. Execution hierarchy

|Ruda concept|目的|
| --- | --- |
|`RudaCount`|Number of workgroups in a launch|
|`RudaDim`|各ワークグループの実行ディメンション|
|`ABSOLUTE_POS`|1 次元要素ごとのカーネル内のグローバル位置|
|`Array<T>`|カーネルでの 1 次元配列アクセス|
|`Tensor<T>`|形状およびストライドのメタデータを使用したカーネル テンソル アクセス|
|`Runtime`|コンパイラ、計算サーバー、およびデバイス タイプを関連付けます|

対応する CUDA の概念については、[互換性ガイド](compatibility.md) を参照してください。パブリック エクスポートは [DSL プレリュード](../../ruda-kernel/src/dsl/prelude.rs) にあります。

## 3. Your first kernel

このカーネルは、完全なホスト プログラムと実行チェックを含む [ptx-runtime example](../../ruda-driver-cuda/examples/ptx_runtime.rs) から来ています。

```rust
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
fn add(a: &Array<f32>, b: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}
```

`ruda_kernel::dsl::prelude::*` を使用してマクロと型をインポートします。 The output length excludes excess tail threads;両方の入力配列には、少なくとも出力と同じ数の要素が含まれている必要があります。

この例では、ワークグループごとに 64 の実行ユニットを起動し、ワークグループのカウントを切り上げます。これは例の構成であり、普遍的に最適なカーネル サイズではありません。

## 4. Memory and arguments

`ComputeClient` を使用してデバイス バッファを作成または割り当て、ハンドルからカーネル引数を構築します。 Distinguish byte counts from element counts:

- `client.empty(size)` takes a size in bytes.
- 例の `ArrayArg::from_raw_parts(handle, count)` では、`count` は配列要素の数です。
- `RudaTensor<R>` は、ストレージ ハンドル、シェイプ、ストライド、dtype、デバイス、および量子化パラメータを保持します。

ハンドルまたはテンソルをクローンしても、基になるデバイス データはコピーされません。レイアウトを変更するには、適切な連続変換、コピー、または変換操作を使用します。メタデータを編集するだけでは、ストレージは再配置されません。

## 5. 送信、リードバック、および同期

カーネルの送信と結果の可用性は別の段階です。ホストの送信が完了しても、デバイスの実行時間や成功は確立されません。

`read_one` はリードバックを待機し、`Result` を返します。 `read_async` は非同期の結果を提供します。 `sync()` は、待機する必要がある未来を返します。 `flush()` はキューに入れられたコマンドを送信します。結果の読み取りに代わるものではありません。

同じデータにアクセスする複数のストリームは、プロデューサーとコンシューマーの依存関係を尊重する必要があります。 `set_stream` は安全ではなく、ホスト変数の有効期間だけではデバイス タスクの完了を確立しません。

## 6. 安全境界線

タイプ、所有権、借用により、ホストのリソースとインターフェイスが制約されます。低レベルのラッパーは、デバイスの実行要件も維持する必要があります。

- 引数のストレージ範囲、dtype、位置合わせ、およびレイアウトはカーネル アクセスと一致します。
- 非同期タスクによって使用されるデータは、完了するまで有効です。
- スレッドおよびストリーム間での共有書き込みは正しく同期されます。
- 呼び出し元が生の引数を構築するか、チェックされていない起動を使用する場合、安全契約は満たされます。

チェックされた起動は、任意のカーネルの完全な安全性を証明するものではありません。この例では、生の引数の構築と起動についての安全性の説明を伴う明示的な `unsafe` ブロックを使用します。 [ランタイム API](runtime-api.md) を参照してください。

## 7. カーネルから計算ライブラリまで

共通操作には [ruBLAS](libraries/rublas.md)、[ruDNN](libraries/rudnn.md)、および [ruPRIM](libraries/ruprim.md) を使用します。カーネル レベルで作業する場合は、入力レイアウト、累積精度、および実行構成を指定します。
