# インストールとクイックスタート

[English](../en/getting-started.md) | [简体中文](../zh/getting-started.md) | **日本語** | [Deutsch](../de/getting-started.md) | [Русский](../ru/getting-started.md)

[ドキュメント](README.md) · [次へ: プログラミング ガイド](programming-guide.md) · [中文](../zh/getting-started.md)

## 1. エントリーポイントを選択してください

|タスク|エントリポイント|
| --- | --- |
|GPU カーネルを書き込みます|`ruda-kernel::dsl` とデバイス ランタイム|
|行列の乗算、FFT、またはリダクションを使用します。|[計算ライブラリ](libraries/README.md)|
|テンソルとフレームワークの操作|[テンソルとフレームワーク](tensor-framework.md)|
|モデルのトレーニングと状態の保存|[トレーニングガイド](training.md)|
|モデルをロードしてテキストを生成するか、画像を処理します|[モデル推論ガイド](model-inference.md)|
|デバイス バックエンドを統合する|[ドライバー API](driver-api.md)|

ソース ワークスペースの Ruda を使用します。

## 2. NVIDIA 環境を準備する

Rust/Cargo、プラットフォーム用のリンカー ツールチェーン、NVIDIA GPU ドライバー、および CUDA ツールキットが必要です。 CUDA バックエンドには、NVRTC とツールキットのビルド依存関係が含まれています。直接 PTX を有効にしても、それらは削除されません。

ソース ルートから環境を確認します。

```powershell
rustc --version --verbose
cargo --version
nvidia-smi
nvcc --version
cargo metadata --no-deps --format-version 1 --offline --locked
```

これらのコマンドは、Ruda をコンパイルしません。 `--offline` では、解決に必要な依存関係をローカルにキャッシュする必要があります。

`CUDA_PATH` を設定して、CUDA ツールキット ルートを選択します。 Windows では、複数のバージョンを含む親ではなく、インストールされているバージョンのディレクトリを指します。 [CUDA インストール パス インターフェイス](../../ruda-driver-cuda/src/lib.rs) を参照してください。

## 3. サンプルをビルドする

ビルド環境の準備ができたら、次を実行します。

```powershell
cargo build --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

`ptx-runtime` の例には、`direct-ptx` が必要です。この機能だけを有効にしても、デフォルトのコンパイラは変更されません。

## 4. コンパイル パスを選択して実行します。

別の PowerShell セッションでパスを 1 つ選択してください。続行する前に、コマンドの失敗を解決してください。

デフォルトの CUDA C++/NVRTC パス:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

直接 PTX パス:

```powershell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

PTX のバージョンは、ターゲット GPU およびドライバーと一致する必要があります。 [PTX バックエンド リファレンス](ptx.md) を参照してください。

この例では、さまざまな長さで FP32 加算を実行し、繰り返し実行される結果とテール センチネルをチェックし、キャッシュ カウンターを出力します。オプションのケースについては、[例とチュートリアル](samples.md) を参照してください。

## 5. 開発を続ける

例のデバイスの選択、データのアップロード、およびカーネルの起動に従って、[プログラミング ガイド](programming-guide.md) を読んでください。ビルド、ドライバーの読み込み、または実行エラーの場合は、[デバッグと診断](debugging.md) を使用して、失敗したステージを特定します。
