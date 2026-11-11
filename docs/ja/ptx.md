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

定義: [PTX モジュール](../../ruda-compiler/src/ptx/mod.rs)、[コンパイラー テスト](../../ruda-compiler/src/ptx/tests.rs)、および [ランタイム バックエンドの選択](../../ruda-driver/cuda/src/compiler_backend.rs)。

## 5. Packed FP4 転送と NVFP4 scaled MMA

`StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2)` は 1 byte に FP4 値を 2 個格納します。直接 PTX はビットを保持する copy／index／select／reinterpret とメモリ転送に対応します。packed byte を単一 FP4 算術値として扱わず、未対応の演算や変換はエラーになります。転送のみでは `sm_120a` は不要です。

`CoopMma::ExecuteScaled` は SM 120、PTX 8.7 以上、`16×8×64` tile、packed E2M1x2 A／B、F32 C／D、E4M3 scale、`scales_factor = 4` のみ受け付けます。`mxf4nvf4.block_scale.scale_vec::4X` を生成し、出力 target は `sm_120a` になります。他のアーキテクチャ、tile、scale 型や係数は拒否され、fragment のレジスタ容量も必要です。

[Packed 処理](../../ruda-compiler/src/ptx/packed.rs)・[scaled MMA](../../ruda-compiler/src/ptx/matrix/scaled.rs)・[転送／配置サンプル](../../ruda-driver/cuda/examples/ptx_packed.rs)。サンプルは出力済み PTX を読み、byte 転送と fragment 座標を調べるもので、NVFP4 MMA 算術は実行しません。

## 6. 自動 lowering と明示的なレイアウト選択

U32/U64 のコンパイル時に既知の非ゼロ定数による除算と剰余は、符号なし入力の全範囲で厳密な整数の強度削減を使用します。2 の累乗にはシフトまたはマスク、条件を満たす他の除数には積の上位ワードとシフト、商が 0 または 1 に限られる大きな除数には比較を使用します。残りは逆数乗算と 1 回の整数補正です。剰余には積の下位ワードの積和を使用できます。入力と出力のレジスタは重複できます。除数が IR 定数の場合に自動適用され、演算子名や固定形状、分子の追加の範囲制限、浮動小数点への変換には依存しません。動的な除数、ゼロ除数、符号付き除算は従来の lowering を維持します。[整数 lowering](../../ruda-compiler/src/ptx/constant_division.rs)を参照してください。

他の自動的な直接 PTX の最適化は次のとおりです。

- 型とターゲットが対応する場合の FP16/BF16 ペア算術と変換。SM100+ と PTX 8.6+ では FP32 の加算、減算、乗算、明示的 FMA もペア化できます。独立した乗算と加算は個別に丸められ、暗黙に低精度を有効にはしません。
- 連続したベクトルメモリアクセスの packed/vector 命令。アラインメントを証明できない場合はチェックとスカラーフォールバックを維持します。SM100+ と PTX 8.8+ のグローバルメモリ命令は最大 32 バイト、それ以外の対応ターゲットは最大 16 バイトです。任意のスライスが整列済みという保証ではありません。
- ブロック形状と制御フローから参加を証明できる場合だけ、warp リダクションの動的な参加マスク処理を省略します。部分 warp や分岐した warp は active-lane パスを維持します。

[行レイアウト API](../../ruda-compiler/src/ptx/row.rs)の `compile_row_variants` は呼び出し側の IR をコンパイルし、`select_row_variant` は呼び出し側の正しさ検証・計時コールバックで候補を選択します。カーネル名から演算子を推定したり、通常のコンパイル中に自動ベンチマークを行ったりはしません。レイアウトと精度は別の判断で、`FastMath::ReducedPrecision` は明示的な指定が必要です。

PTX は最終マシンコードではありません。NVIDIA アセンブラによって命令数やレジスタ割り当てが変わり、命令数が少なくても高速とは限りません。対象ワークロードのアセンブル結果とデバイス上の時間を比較してください。コンパイルやアセンブルの成功は性能結果ではありません。デフォルトの NVRTC 選択と起動・引数の契約は変更されません。
