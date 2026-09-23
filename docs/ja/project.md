# Ruda — Rust 高性能計算

![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange?logo=rust&logoColor=white)
![言語](https://img.shields.io/github/languages/top/shuqi2077/RUDA)
![フォーク](https://img.shields.io/github/forks/shuqi2077/RUDA?style=flat)
![Issue](https://img.shields.io/github/issues/shuqi2077/RUDA)
![最終コミット](https://img.shields.io/github/last-commit/shuqi2077/RUDA?display_timestamp=committer)

[English](../../README.md) | [简体中文](../zh/project.md) | **日本語** | [Deutsch](../de/project.md) | [Русский](../ru/project.md)

Ruda は Rust の高性能計算ライブラリです。GPU カーネル、コンパイラ、ランタイムから、数学計算、テンソル、モデルまでを網羅するソフトウェアスタックを構築しています。

Ruda は CUDA C++ のコンパイル経路を維持しつつ、PTX、HIP、独自 ISA を対象とする Rust のコンパイル・実行経路を構築しています。低レベルの制御された `unsafe` のカプセル化と、高レベルの Rust の型システム、所有権、借用を組み合わせ、低レベルの性能制御と高レベルのメモリ安全性を両立します。

## クイックスタート

Git、Rust/Cargo、リンカーツールチェーン、NVIDIA GPU とドライバ、CUDA Toolkit が必要です。インストールの詳細は[環境構築](getting-started.md)を参照してください。

### 公開済みクレートの利用

アプリケーションの `Cargo.toml` に [CUDA バックエンド](https://crates.io/crates/ruda-driver-cuda)を追加します。

```toml
[dependencies]
ruda-driver-cuda = { version = "0.1", features = ["direct-ptx"] }
```

以下のサンプルはソースディレクトリから実行します。

### クローン

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
```

### GPU カーネルの実行

シェルで直接 PTX コンパイラを選択します。

```sh
# Bash
export RUDA_CUDA_COMPILER=ptx
export RUDA_PTX_VERSION=8.0
```

```powershell
# PowerShell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
```

続いてサンプルをビルドして実行します。

```sh
cargo run --release --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

このサンプルは GPU 上で FP32 加算を実行し、`PASS` 行とコンパイルキャッシュのカウンタを表示します。GPU とドライバが対応する [PTX バージョン](ptx.md)を選択してください。

### ruLLM によるテキスト生成

ローカルの Qwen3.5-0.8B モデルを `./models/qwen35` に配置するか、以下のパスをモデルのディレクトリに置き換えてください。モデルファイルは付属しません。[モデルの準備](model-inference.md#ローカルモデルの準備)を参照してください。

```sh
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "The capital of France is" 8 1
```

このサンプルは生成されたテキストとトークン ID を表示します。代わりに CUDA C++ / NVRTC 経路を使う場合は、いずれのサンプルも実行前に `RUDA_CUDA_COMPILER` を `nvrtc` に設定してください。

### ネイティブ PyTorch バックエンドの利用

`ruda-torch` は単一の NVIDIA GPU 上に PyTorch デバイス `ruda:0` を登録します。PyTorch、setuptools、C++20 コンパイラを用意し、上記の PTX 環境設定を使ってリポジトリのルートで次を実行してください。Windows では x64 MSVC 開発者シェルを使用します。

```sh
cargo build --locked -p ruda-torch-native
python -m pip install --no-build-isolation --no-deps -e ./ruda-torch/python
```

既定のローダーはこの debug ビルドを自動検出します。release ビルドや別の場所のライブラリを使う場合は、そのパスを `RUDA_TORCH_LIBRARY` に設定してください。Rust ライブラリと C++ 拡張は両方とも **ABI 9** が必要で、同時に再ビルドします。

```python
import torch
import ruda_torch

x = torch.arange(4, dtype=torch.float32).to("ruda:0")
print((x + x).cpu())
```

ビルド済み Windows wheel は、成功した [RUDA Torch Windows build](https://github.com/shuqi2077/RUDA/actions/workflows/ruda-torch-windows.yml) のアーティファクトから取得できます。wheel アーティファクトを展開し、中の `.whl` ファイルを `python -m pip install --no-deps` でインストールしてください。wheel はネイティブ DLL を含み、Windows x64、CPython 3.13、PyTorch `2.13.0+cu130` 向けです。対応する PyTorch を先にインストールしてください。アーティファクトの保存期間は 7 日です。`ruda-torch-native` はソースからビルドするコンポーネントで、crates.io パッケージではありません。

## スタックの構成

1 つのリポジトリに、責務が明確な複数のクレートを収めています。分野別ライブラリから上位フレームワークまで、スタックを階層化し、一体として開発しています。

| 層 | コンポーネント |
| --- | --- |
| 共通契約 | `ruda-core` |
| コンパイルとカーネル | `ruda-compiler`、`ruda-kernel`、マクロコンポーネント |
| ランタイムとドライババックエンド | `ruda`、`ruda-driver-cuda/cpu/wgpu/hip` |
| 分野別ライブラリ | ruBLAS、ruDNN、ruPRIM、ruFFT、ruRAND、ruSPARSE |
| 集合通信 | ruCCL、`ruda-communication` |
| テンソルとフレームワーク | `ruda-tensor*`、`ruda-autodiff`、`ruda-fusion` |
| PyTorch 統合 | `ruda-torch-native`（Rust）、`ruda_torch`（Python） |
| モデルとデータ | `ruda-model`、`ruda-nn`、`ruda-optim`、`ruda-store`、`ruda-dataset` |

## ネイティブ GPU 推論

- **演算子:** ネイティブ PyTorch の行列演算は ruBLAS を使用します。FP16/BF16 ストレージを維持する計算経路、最終軸の融合 LayerNorm/RMSNorm、ワープ並列 Softmax/リダクションにより、中間テンソルと個別のカーネル投入を削減します。
- **ページ化 GQA と MLA:** ruDNN の公開カーネルは物理 KV ページを直接読み、可変長の prefill/decode を処理します。`ruda_torch.PagedAttentionPlan` は `splits=1..32`、FP32 の部分結果の結合、ワークスペースの再利用に対応し、既定値は `splits=1` です。共有キャッシュへの書き込みではコピーオンライト保護を維持します。
- **MoE:** グループ化 sigmoid ルーティングとセグメント化エキスパート行列積は、デバイス上のエキスパートオフセットを使用します。FP16/BF16 Tensor Core 経路は明示的な選択が必要で、既存のエキスパート API は既定でスカラー GPU 戦略を使います。
- **ストリームとイベント:** `ruda_torch.Stream`、`Event`、`record_stream` はネイティブランタイムに接続します。投入は既定で同期的です。最初のネイティブ処理投入前に `RUDA_TORCH_ASYNC=1` を設定すると非同期投入を有効にできます。明示的な同期とホストへの読み戻しは引き続き完了を待ちます。

ページ化 Attention には、同じデータ型・デバイス・実行キュー上の連続した FP32/FP16/BF16 テンソルが必要です。順伝播のみで、任意の外部マスクや量子化 KV キャッシュには対応しません。MLA/MoE は再利用可能なコンポーネントであり、完全なモデルアダプターには射影、位置エンコーディング、ルーティングパラメーター、キャッシュ所有権の管理が必要です。

## ハードウェアへの経路

- **NVIDIA GPU:** 既定のコンパイル経路は CUDA C++ → NVRTC → PTX です。IR → PTX の直接生成も明示的に選択できます。どちらも NVIDIA ドライバ経由で実行します。
- **その他の実行バックエンド:** CPU、WGPU、HIP のバックエンドソースがあります。対応範囲は[互換性](compatibility.md)を参照してください。

## 詳細と貢献

- [Ruda ドキュメント](README.md): クイックスタート、プログラミングガイド、コンパイラ、API リファレンス、計算ライブラリのマニュアル。
- [NVIDIA デモ](getting-started.md): サンプルと実行要件を確認できます。
- [貢献ガイド](CONTRIBUTING.md): 演算、コンパイラ、ランタイム、フレームワークへの貢献。

Rust、GPU カーネル、コンパイラ、高性能計算に関心のある方は、このスタックをさらに発展させ、高速化する取り組みにご参加ください。

## 由来とライセンス

[サードパーティに関する通知](../../THIRD_PARTY_NOTICES.md)

プロジェクトがライセンスを付与する権利を持つ Ruda 独自のソフトウェアコードは、[Apache License 2.0](../../LICENSE) で提供されます。サードパーティのファイルには元のライセンスが適用され、ルートのライセンスは移行されたコンポーネントの `MIT OR Apache-2.0` 宣言を上書きしません。
