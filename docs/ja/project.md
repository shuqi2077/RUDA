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
cargo run --release --locked -p ruLLM --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "The capital of France is" 8 1
```

このサンプルは生成されたテキストとトークン ID を表示します。代わりに CUDA C++ / NVRTC 経路を使う場合は、いずれのサンプルも実行前に `RUDA_CUDA_COMPILER` を `nvrtc` に設定してください。

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
| モデルとデータ | `ruda-model`、`ruda-nn`、`ruda-optim`、`ruda-store`、`ruda-dataset` |

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
