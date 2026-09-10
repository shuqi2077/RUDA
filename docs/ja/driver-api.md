# ドライバー API とバックエンド

[English](../en/driver-api.md) | [简体中文](../zh/driver-api.md) | **日本語** | [Deutsch](../de/driver-api.md) | [Русский](../ru/driver-api.md)

[ドキュメント](README.md) · [ランタイム API](runtime-api.md) · [互換性](compatibility.md) · [中文](../zh/driver-api.md)

Ruda のドライバクレートは、汎用ランタイムの契約を実行バックエンドに接続します。本ガイドは Rust バックエンドのエントリポイントを説明するもので、CUDA Driver API 関数をそのまま置き換えるものではありません。

## 1. バックエンドクレート

|クレート|実行バックエンド|
| --- | --- |
|`ruda-driver-cuda`|NVIDIA CUDA ドライバーとコンパイル パス|
|`ruda-driver-cpu`|CPU|
|`ruda-driver-wgpu`|WGPU|
|`ruda-driver-hip`|HIP|

## 2. NVIDIA デバイスを選択します

`ruda_driver_cuda::CudaDevice` は、パブリック `index: usize` フィールドを通じてデバイスを選択します。デフォルトは 0 です。 `CudaRuntime` は、`Runtime` 特性を実装します。

次のコマンドを使用して、デフォルトのデバイスのクライアントを取得します。

```rust
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::dsl::Runtime;

let client = CudaRuntime::client(&CudaDevice::default());
```

完全な実行可能な例については、[ptx-runtime](../../ruda-driver-cuda/examples/ptx_runtime.rs) を参照してください。デバイス インデックスは、現在のマシンの列挙位置を識別するものであり、マシン間での安定した ID や列挙順序の変更を識別するものではありません。

## 3. 設定

`RuntimeOptions` には、メモリ管理用の `memory_config` が含まれています。 `CudaCompiler` および `CudaComputeKernel` は、CUDA C++ コンパイル チェーンの型エイリアスです。直接 PTX を有効にしても、その意味は変わりません。

[コンパイラ ガイド](compiler-guide.md) の説明に従って、`RUDA_CUDA_COMPILER` でコンパイル パスを選択します。 `install::cuda_path()`、`install::include_path()`、および `install::cccl_include_path()` は、ツールキットのパスを見つけます。

## 4. 外部依存関係

PTX を直接生成すると、カーネルの CUDA C++/NVRTC コンパイル ステップがバイパスされます。実行には依然として NVIDIA ドライバーが必要であり、クレートは NVRTC の依存関係を保持します。

インターフェイスは、任意の外部 CUDA コンテキスト、ストリーム、または RAW デバイス ポインターの採用を保証しません。言語間の統合では、所有権、実行の依存関係、エラーの伝播を考慮する必要があります。 CUDA バックエンドだけでは、完全な ABI 互換性は提供されません。

## 5. 別のバックエンドを統合する

バックエンドは、`Runtime` を使用して、デバイス、コンパイラー、およびコンピューティング サーバーを関連付けます。上位層は、`ComputeClient` を通じてコントラクトにアクセスします。計算ライブラリを統合する前に、ストレージ、コンパイル エラー、同期、機能クエリのセマンティクスを確立します。

ソース エントリ ポイント: [CUDA エクスポート](../../ruda-driver-cuda/src/lib.rs)、[デバイス タイプ](../../ruda-driver-cuda/src/device.rs)、[ランタイム実装](../../ruda-driver-cuda/src/runtime.rs)、および [ランタイム特性](../../ruda/src/runtime/backend.rs)。
