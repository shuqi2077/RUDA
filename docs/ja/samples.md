# の例とチュートリアル

[English](../en/samples.md) | [简体中文](../zh/samples.md) | **日本語** | [Deutsch](../de/samples.md) | [Русский](../ru/samples.md)

[ドキュメント](README.md) · [クイックスタート](getting-started.md) · [中文](../zh/samples.md)

## 1. NVIDIA ランタイムの例

エントリ ポイント: [ptx-runtime](../../ruda-driver-cuda/examples/ptx_runtime.rs)。

基本的なケースでは、FP32 加算を各長さ (1、63、64、65、257) で 2 回実行します。すべての結果と 16 個の末尾センチネル値をチェックします。デバイスの選択、アップロード、引数バインド、テール バウンド、リードバック、および繰り返し実行を示します。

バックエンドの構築と選択については、[クイックスタート](getting-started.md) を参照してください。

## 2. 対象となるケース

これらの引数はコマンドの `--` の後に配置します。たとえば、次のようにしてテンソルのケースを実行します。

```powershell
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --tensor
```

|引数|の動作|ソース例|
| --- | --- | --- |
|`--tensor`|Tensor メタデータとレイアウトのチェック|[tensor.rs](../../ruda-driver-cuda/examples/ptx_runtime/tensor.rs)|
|`--shared`|共有メモリのチェック|[shared.rs](../../ruda-driver-cuda/examples/ptx_runtime/shared.rs)|
|`--half`|FP16/BF16 チェック|[half_precision.rs](../../ruda-driver-cuda/examples/ptx_runtime/half_precision.rs)|
|`--bitwise`|ビット単位の動作チェック|[bitwise.rs](../../ruda-driver-cuda/examples/ptx_runtime/bitwise.rs)|
|`--shared-over-limit`|共有メモリ制限診断|[shared.rs](../../ruda-driver-cuda/examples/ptx_runtime/shared.rs)|
|`--expect-cold`|ディスク キャッシュ ヒットなしでコンパイルが行われたことをアサートします|[メイン エントリ ポイント](../../ruda-driver-cuda/examples/ptx_runtime.rs)|
|`--expect-warm`|再コンパイルせずにディスク キャッシュ ヒットをアサートします|[メイン エントリ ポイント](../../ruda-driver-cuda/examples/ptx_runtime.rs)|

`--shared-over-limit` は別の早期リターン分岐を実行します。コールド/ウォーム キャッシュ チェックと組み合わせないでください。

## 3. コールド キャッシュとウォーム キャッシュ

コンパイル パスごとに個別の新しいキャッシュ ディレクトリを使用します。同じパスのコールド実行とウォーム実行にそのディレクトリを再利用します。

[はじめに](getting-started.md) の説明に従ってコンパイル パスを選択した後、同じ PowerShell セッションで次のコマンドを実行します。

```powershell
$env:RUDA_PTX_TEST_CACHE = 'target/ptx-example-cache-' + [guid]::NewGuid().ToString('N')
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --expect-cold
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --expect-warm
```

コンパイラ、入力引数、およびキャッシュ パスを実行間で変更しないようにします。

## 4. 計算ライブラリの例

サンプルを実行する前に、[ビルド環境](getting-started.md) を準備してください。

|タスク|コマンド|出力チェック|
| --- | --- | --- |
|FP32 CSR 行列ベクトル乗算|`cargo run --locked -p ruSPARSE --features cuda --example csrmv`|`[7.0, 2.0, 18.5]` を読み戻してチェックします|
|CUDA リング AllReduce|`cargo run --locked -p ruCCL --features cuda --example all_reduce`|GPU 0、257 要素、合計/平均、および入力保存の 4 つの論理ランク|

## 5. トレーニングとモデル推論

- [トレーニングと状態の保存](training.md): 前方、後方、勾配の累積、およびトレーニングの記録。
- [モデルの読み込みと推論](model-inference.md): Qwen2/Qwen3.5 のサンプル コマンド、チャット、サンプリング、AWQ、および画像入力。
