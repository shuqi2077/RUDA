# PTX バックエンドリファレンス

[English](../en/ptx.md) | [简体中文](../zh/ptx.md) | **日本語** | [Deutsch](../de/ptx.md) | [Русский](../ru/ptx.md)

[ドキュメント](README.md) · [コンパイラ ガイド](compiler-guide.md) · [例](samples.md) · [中文](../zh/ptx.md)

## 1. 範囲

`ruda_compiler::ptx` は、Ruda カーネル IR から PTX テキストを生成します。これは、任意の PTX プログラムのインタープリターでも、最終的な NVIDIA マシン コードのジェネレーターでもありません。

スタンドアロン コンパイルに対して `ruda-compiler/ptx` を有効にします。 CUDA ランタイムを介してこのパスを選択するには、`ruda-driver-cuda/direct-ptx` を有効にします。

## 2. ターゲットの種類

|タイプ/フィールド|意味|
| --- | --- |
|`PtxTarget::version: (u32, u32)`|PTX メジャー/マイナー バージョン|
|`PtxTarget::sm: u32`|SM ターゲット|
|`PtxCompilationOptions::target: Option<PtxTarget>`|明示的なターゲット構成|
|`PtxCompiler`|パブリック コンパイラ特性を実装するダイレクト バックエンド|

`target` が存在しない場合、コンパイルは検証エラーを返します。スタンドアロン コンパイラーは、ホスト マシンから GPU アーキテクチャを推論しません。

環境変数の解析では、メジャー バージョンが 6 以上、マイナー バージョンが 9 以下の `major.minor` を受け入れます。構文の妥当性だけでは、そのバージョンのドライバーまたはジェネレーターのサポートは確立されません。

## 3. コンパイル出力

`PtxKernel` には次の内容が含まれます。

- `source`: PTX テキスト。
- `entrypoint`: エントリ ポイント名。
- `ruda_dim`: 元のカーネルのワークグループ ディメンション。
- `shared_memory_bytes`: 共有メモリが必要です。
- `dynamic_metadata_index`: 必要な場合の動的メタデータ ポインター引数の位置。

実行層を呼び出すときに、引数のレイアウト、エントリ ポイント、および共有メモリの要件を保持します。このテキストだけでは完全な立ち上げ契約は含まれていません。

## 4. エラー処理

サポートされていない IR は `CompilationError::UnsupportedInstruction` を返します。無効な構成または構造は、`CompilationError::Validation` を返します。診断には、`Direct PTX:` プレフィックスが含まれます。バックエンドは自動的に NVRTC にフォールバックしません。

定義: [PTX モジュール](../../ruda-compiler/src/ptx/mod.rs)、[コンパイラー テスト](../../ruda-compiler/src/ptx/tests.rs)、および [ランタイム バックエンドの選択](../../ruda-driver-cuda/src/compiler_backend.rs)。
