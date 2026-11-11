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

機能により、どのコードが構築されるかが決まります。環境変数は実行時にパスを選択します。これらは別個のコントロールです。 [CUDA マニフェスト](../../ruda-driver/cuda/Cargo.toml) および [コンパイラー マニフェスト](../../ruda-compiler/Cargo.toml) を参照してください。

## 3. 環境変数

|変数 |行動 |
| --- | --- |
| `RUDA_CUDA_COMPILER` | `nvrtc` または `ptx` を受け入れます。設定されていない場合、デフォルトは nvrtc になります。|
| `RUDA_PTX_VERSION` |直接 PTX には明示的な `major.minor` バージョンが必要です。|
| `CUDA_PATH` | CUDA ツールキットのインストール ルート |
| `RUDA_PTX_TEST_CACHE` | ptx-runtime サンプルだけが読み取るキャッシュディレクトリの上書き設定 |

不明なコンパイラ値によりエラーが発生します。 `direct-ptx` を有効にせずに `ptx` を選択すると、NVRTC に切り替えるのではなく、エラーが発生します。 [compiler_backend.rs](../../ruda-driver/cuda/src/compiler_backend.rs)を参照してください。

## 4. ターゲットとキャッシュ

スタンドアロンの直接 PTX コンパイルには、明示的な PTX バージョンと SM ターゲットが必要です。 PTX は命令セットのバージョンを識別します。 SM はターゲット アーキテクチャを識別します。これらは交換可能ではありません。

CUDA ドライバーの直接 PTX キャッシュ名前空間には、バックエンド識別子、SM、および PTX バージョンが含まれており、NVRTC キャッシュとは別のものです。

## 5. コンパイルエラー

サポートされていない IR、引数メタデータ、またはターゲット条件によりエラーが発生します。ダイレクト PTX は、コンパイル パスを自動的に切り替えるのではなく、サポートされていない操作を報告します。 [デバッグ](debugging.md)を参照してください。

`ruda-compiler` には、WGSL、SPIR-V、および MLIR モジュールも含まれています。

## 6. Triton フロントエンド

ソースビルド専用の `ruda-triton` は、Triton TTIR → SIMT 記述 → Ruda IR → 直接 PTX を実装します。`cargo build --locked --release -p ruda-triton` でコンパイラーを作り、pip で `./ruda-triton/python` をインストールします。宣言された依存関係は Python 3.10+、Triton 3.8+、PyTorch です。`RUDA_TRITON_PTX` を `target/release/ruda-triton-ptx`（Windows は `.exe`）へ設定するか PATH に追加します。

Triton の import 前に `TRITON_DEFAULT_BACKEND=ruda` を設定します。既定の `RUDA_TRITON_DEVICE=cuda` は `torch.cuda` ストレージと現在の CUDA stream を使います。`ruda` は `ruda_torch` PrivateUse1 ストレージと現在の RUDA stream を使い、起動前に同 stream の RUDA キューを flush します。CUDA と RUDA のポインターは交換できません。

Triton が対象で Gluon は対象外です。未対応 TTIR は lowering エラーになります。Rust Kernel DSL とは別経路で、すべての Triton kernel への対応を意味しません。GX1 モードにはこの workspace 外の独立ライブラリが必要です。

[依存関係](../../ruda-triton/python/pyproject.toml)・[コンパイラー](../../ruda-triton/python/ruda_triton/compiler.py)・[デバイス／stream](../../ruda-triton/python/ruda_triton/driver.py)。

## 7. 直接 PTX の最適化範囲

直接バックエンドは、条件を満たす U32/U64 の定数除算・剰余、ベクトル算術・変換、整列メモリアクセス、warp リダクションを自動的に lowering します。[PTX リファレンス](ptx.md)に、厳密な整数の意味、SM/PTX の条件、アラインメントのフォールバック、参加マスクの規則を記載しています。

協調行レイアウトの探索は呼び出し側の IR に対する明示的 API であり、自動演算子認識や実行時の自動チューニングではありません。低精度は別の明示的オプションです。PTX 命令数が少なくても高速化を証明しないため、レジスタ使用量とデバイス上の時間を確認してください。デフォルトの `nvrtc` 選択は変わりません。
