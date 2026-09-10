# コンパイラガイド

[English](../en/compiler-guide.md) | [简体中文](../zh/compiler-guide.md) | **日本語** | [Deutsch](../de/compiler-guide.md) | [Русский](../ru/compiler-guide.md)

[ドキュメント](README.md) · [PTX リファレンス](ptx.md) · [プログラミング ガイド](programming-guide.md) · [中文](../zh/compiler-guide.md)

## 1. コンパイルパイプライン

汎用カーネル経路は `ruda-kernel::dsl` のマクロと型から始まり、Kernel IR を生成し、バックエンドで低位表現への変換とコード生成を行います。`ruda-compiler` にコンパイラ実装があり、デバイスドライバがその出力を実行環境に渡します。

NVIDIA は 2 つのコンパイル パスを提供します。

|選択 |カーネルコンパイルパイプライン |
| --- | --- |
| `nvrtc` (デフォルト) | Rust カーネル フロントエンド → IR → CUDA C++ → NVRTC → PTX |
| `ptx` | Rust カーネル フロントエンド → IR → PTX |

どちらも NVIDIA ドライバーを通じて実行されます。 CUDA C++ コンパイル パスは、任意の C++ プロジェクトをインポートしたり、CUDA ソースの完全な互換性を提供したりするためのインターフェイスではありません。

## 2. Cargo feature

|コンポーネント/機能 |目的 |
| --- | --- |
| `ruda-kernel/frontend` |カーネル DSL フロントエンド |
| `ruda-kernel/lowering-cpp` | C++ への低位化の統合 |
| `ruda-compiler/cpp` | C++ バックエンドの実装 |
| `ruda-compiler/ptx` |直接 PTX コンパイラ |
| `ruda-driver-cuda/direct-ptx` | CUDA バックエンドでの直接 PTX 選択を有効にします。|

機能により、どのコードが構築されるかが決まります。環境変数は実行時にパスを選択します。これらは別個のコントロールです。 [CUDA マニフェスト](../../ruda-driver-cuda/Cargo.toml) および [コンパイラー マニフェスト](../../ruda-compiler/Cargo.toml) を参照してください。

## 3. 環境変数

|変数 |行動 |
| --- | --- |
| `RUDA_CUDA_COMPILER` | `nvrtc` または `ptx` を受け入れます。設定されていない場合、デフォルトは nvrtc になります。|
| `RUDA_PTX_VERSION` |直接 PTX には明示的な `major.minor` バージョンが必要です。|
| `CUDA_PATH` | CUDA ツールキットのインストール ルート |
| `RUDA_PTX_TEST_CACHE` | ptx-runtime サンプルだけが読み取るキャッシュディレクトリの上書き設定 |

不明なコンパイラ値によりエラーが発生します。 `direct-ptx` を有効にせずに `ptx` を選択すると、NVRTC に切り替えるのではなく、エラーが発生します。 [compiler_backend.rs](../../ruda-driver-cuda/src/compiler_backend.rs)を参照してください。

## 4. ターゲットとキャッシュ

スタンドアロンの直接 PTX コンパイルには、明示的な PTX バージョンと SM ターゲットが必要です。 PTX は命令セットのバージョンを識別します。 SM はターゲット アーキテクチャを識別します。これらは交換可能ではありません。

CUDA ドライバーの直接 PTX キャッシュ名前空間には、バックエンド識別子、SM、および PTX バージョンが含まれており、NVRTC キャッシュとは別のものです。

## 5. コンパイルエラー

サポートされていない IR、引数メタデータ、またはターゲット条件によりエラーが発生します。ダイレクト PTX は、コンパイル パスを自動的に切り替えるのではなく、サポートされていない操作を報告します。 [デバッグ](debugging.md)を参照してください。

`ruda-compiler` には、WGSL、SPIR-V、および MLIR モジュールも含まれています。
