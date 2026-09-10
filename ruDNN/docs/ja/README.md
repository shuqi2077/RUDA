# ruDNN

[English](../../README.md) | [简体中文](../zh/README.md) | **日本語** | [Deutsch](../de/README.md) | [Русский](../ru/README.md)

**英語** | [简体中文](../zh/README.md)

Ruda のニューラル ネットワーク オペレーター。

- Cargo パッケージ: `ruDNN`
- Rust クレート: `rudnn`

## feature

| feature |操作|
| --- | --- |
|`tensor-attention`| Attention |
|`tensor-convolution`|畳み込み|
|`tensor-normalization`|LayerNorm、RMSNorm、およびソフトマックス|
|`tensor-moe`|MoE ルーティング、ディスパッチ、エキスパート計算|
|`tensor-gated-delta`|ゲートデルタ計算|
|`pooling`|プーリング|
|`interpolation`|補間|
|`grid-sample`|グリッドサンプリング|
|`ctc`|CTC 損失|

## クイック スタート

RUDA ワークスペースからビルドします。

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruDNN --features tensor-normalization
```

## ドキュメント

- [ユーザーガイド](../../../docs/ja/libraries/rudnn.md)
- [環境設定](../../../docs/ja/getting-started.md)
- [Cargo 機能](../../Cargo.toml) · [モジュール エクスポート](../../src/lib.rs)

## ruDNN ユーザーガイド

[計算ライブラリ](../../../docs/ja/libraries/README.md) · [ruBLAS](../../../docs/ja/libraries/rublas.md) · [テンソルとフレームワーク](../../../docs/ja/tensor-framework.md) · [中文](../zh/README.md)

### 1. 概要と特徴

ruDNN はニューラル ネットワーク操作を提供します。 Cargo パッケージは `ruDNN`、Rust クレートは `rudnn` です。

| feature |オペレーション|
| --- | --- |
|`tensor-attention`|テンソル アテンション|
|`tensor-convolution`|テンソル畳み込み|
|`pooling`、`interpolation`|プーリングと補間|
|`grid-sample`、`ctc`|グリッド サンプリングと CTC|
|`tensor-moe`|デバイス MoE ルーティングと専門家による計算|
|`tensor-normalization`|モデル層の汎用デバイス正規化|
|`tensor-gated-delta`|Qwen3.5 などのハイブリッド アーキテクチャ向けのゲートデルタ計算|

アテンションとコンボリューションには、それぞれ対応する自動調整機能があります。 [Cargo.toml](../../Cargo.toml) および [モジュール エクスポート](../../src/lib.rs) を参照してください。

### 2. Attention と畳み込み

#### Attention

`rudnn::attention::tensor::attention` は、クエリ、キー、値、オプションのマスク、オプションの attn_bias、AttentionModuleOptions、および AttentionStrategy を受け取ります。デバイス テンソルまたは AttentionSetupError を返します。

戦略には FlashBlackboxAccelerated、FlashUnit、Fallback、および対応 feature が有効な場合の Autotune があります。自動チューニングが無効なら既定は Fallback、有効なら Autotune です。Fallback は同じデバイス上で複数のカーネルを使い、CPU バックエンドは使いません。

レイアウト、マスク、精度を選択した戦略に合わせます。 [アテンションインターフェイス](../../src/attention/tensor/base.rs)を参照してください。

#### コンボリューション

`rudnn::convolution::tensor::conv_forward` は、入力、重み、オプションの bias、ConvOptions<N>、および ConvStrategy を受け取ります。デバイス テンソルまたは ConvSetupError を返します。実行のために入力をチャネル最後のレイアウトに変換し、出力を逆変換します。 `conv_forward_nhwc` は、チャネル最後のレイアウトを直接使用します。

戦略には、直接、ImplicitGemm、およびオプションの Autotune が含まれます。エントリ ポイントは、3 次元 F32 畳み込みに Direct を使用します。 ImplicitGemm が選択されている場合、グループ化コンボリューションでもダイレクトが使用されます。したがって、戦略パラメーターは、常に 1 つのアルゴリズムを保持するという厳密な要求ではありません。

同じモジュールで conv_data_backward と conv_weight_backward が提供されます。各方向の形状、オプション、施工条件をご確認ください。 [コンボリューションインターフェイス](../../src/convolution/tensor/base.rs)を参照してください。

### 3. MoE ワークフロー

`rudnn::moe` は、次のローカル計算シーケンスを提供します。

入力 logits → softmax/top-k → コンパクトな振り分け → SwiGLU エキスパート → 重み付き結合。

|API|目的|
| --- | --- |
|`route(logits, RoutingOptions)`|は RoutingPlan を作成します|
|`RoutingPlan::expert_indices()`、`weights()`|選択した専門家と重み付けにアクセスします|
|`RoutingPlan::dispatch(input)`|エキスパートによるトークンのディスパッチ|
|`SwiGluExperts::new(gate, up, down)`|bias フリーのエキスパート ウェイトを作成します|
|`SwiGluExperts::forward_dispatched`|ディスパッチされたトークンを計算します|
|`DispatchedTokens::combine`|トークンの順序を復元し、重み付けされた結果を結合します|
|`SwiGluExperts::forward(input, logits, options)`|完全なローカル転送シーケンスを実行します|

これらのインターフェイスは `RudaTensor<R>` を使用します。計算エントリ ポイントは、無効なシェイプ、dtype、またはその他の引数に対して `Result` と `MoeError` を返します。

### 4. ルーティング契約

Logits の形状は [T, E] で、非量子化の F32、F16、BF16 を使います。`RoutingOptions` は `top_k: usize` と `renormalize: bool` を含み、1 ≤ top_k ≤ E が必要です。

Softmax は、top-k を選択する前に、FP32 のすべてのエキスパートを計算します。同点の場合、エキスパート ID が低いほど有利になります。再正規化を有効にすると、選択した重みが再正規化され、ロジット dtype にキャストされます。 NaN、正の無限大、または負の無限大のみを含む行は、一様分布を使用するのではなく、NaN の重みを保持します。

選択されたエキスパート インデックスとウェイトは両方とも形状 [T、top_k] です。インデックスはU32を使用します。

### 5. エキスパートの重みと振り分け

入力トークンの形状は [T, H]、gate と up は [E, I, H]、down は [E, H, I] です。重みは同じ浮動小数点 dtype とデバイスを使う必要があります。エキスパート数と各次元はルーティングおよび入力に一致していなければなりません。

振り分け処理は容量制限を守るためにトークンを破棄しません。エキスパート内のアトミックな割り当て順序は固定されず、保存した対応表でトークン順序を復元します。順伝播のエントリポイントは計算済み logits を受け取り、モデルのゲート射影や重みファイルの読み込みは行いません。

ソース: [ルーティング](../../src/moe/routing.rs)、[ディスパッチと結合](../../src/moe/dispatch.rs)、[専門家](../../src/moe/experts.rs)、および [テスト](../../src/moe/tests.rs)。

### 6. MoE の呼び出し

`ruDNN` 依存関係で `tensor-moe` を有効にします。上記のレイアウトでデバイス テンソルを準備し、エキスパート ウェイトを作成して、順方向操作を実行します。

```rust
use rudnn::moe::{RoutingOptions, SwiGluExperts};

let experts = SwiGluExperts::new(gate, up, down)?;
let options = RoutingOptions {
    top_k: 2,
    renormalize: true,
};
let output = experts.forward(input, logits, options)?;
```

これにより、トークンごとに 2 人のエキスパートが選択されるため、`E` は少なくとも 2 である必要があります。 `input` の形状は `[T, H]` で、`logits` の形状は `[T, E]` で、`output` の形状は `[T, H]` です。重みオブジェクトを再作成せずに、後続の入力バッチ全体で `experts` を再利用します。

ルーティングを検査するか、ステージ間に独自の処理を挿入するには、ステージを個別に呼び出します。

```rust
use rudnn::moe::route;

let routing = route(logits, options)?;
let selected_experts = routing.expert_indices();
let selected_weights = routing.weights();
let dispatched = routing.dispatch(input)?;
let expert_output = experts.forward_dispatched(&dispatched)?;
let output = dispatched.combine(expert_output)?;
```

これらは代替形式です。 2 番目は、`input` と `logits` の新しいバッチで始まります。 `combine` は、ディスパッチ マッピングを使用してトークンの順序を復元し、ルーティングの重みを使用してエキスパートの出力をマージします。

形状、dtype、またはデバイスの不一致により、`MoeError` が返されます。ホスト上で結果を読み取るには、`ruda_kernel::tensor::readback::into_data_sync(output)` を使用します。これは、デバイスの結果を待機して、`TensorData` を返します。

### 7. LayerNorm、RMSNorm、および Softmax

`tensor-normalization` を有効にし、これらの関数を `rudnn::normalization` からインポートします。すべては最終入力軸上で動作し、形状を保持します。

|関数|入力|パラメータ|
| --- | --- | --- |
|`layer_norm(input, gamma, beta, epsilon)`|F32/F16/BF16|F32 ベクトル `gamma`、オプションの F32 ベクトル `beta`|
|`rms_norm(input, gamma, epsilon)`|F32/F16/BF16|F32 ベクトル `gamma`|
|`softmax_last_axis(input)`|F32|追加パラメータはありません|

入力は空ではない最終軸で量子化されていない必要があります。最終軸長 H の場合、`gamma` および `beta` は `[H]` の形状を持ち、量子化されておらず、入力デバイスを共有する必要があります。 `epsilon` は有限かつ正でなければなりません。 BF16/F16 入力の場合でも、アフィン パラメーターは F32 のままです。統計とアフィン演算は FP32 を使用し、出力は入力 dtype にキャストされます。

```rust
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use rudnn::normalization::{NormalizationError, layer_norm, rms_norm, softmax_last_axis};

fn normalize<R: Runtime>(
    input: RudaTensor<R>,
    gamma: RudaTensor<R>,
    beta: Option<RudaTensor<R>>,
    epsilon: f32,
) -> Result<RudaTensor<R>, NormalizationError> {
    layer_norm(input, gamma, beta, epsilon)
}

fn normalize_rms<R: Runtime>(
    input: RudaTensor<R>,
    gamma: RudaTensor<R>,
    epsilon: f32,
) -> Result<RudaTensor<R>, NormalizationError> {
    rms_norm(input, gamma, epsilon)
}

fn probabilities<R: Runtime>(
    logits: RudaTensor<R>,
) -> Result<RudaTensor<R>, NormalizationError> {
    softmax_last_axis(logits)
}
```

### 8. Gated-delta のプリフィルと逐次更新

`tensor-gated-delta` を有効にします。チャンク シーケンスのプリフィルには `chunk_gated_delta_rule(input, chunk_size)` を使用し、トークンごとの繰り返しには `gated_delta_rule(input)` を使用します。どちらも `GatedDeltaInput<R>` を使用します。

|フィールド| 形状／型 |
| --- | --- |
|`query`、`key`|`[B, H, T, K]`、F32/F16/BF16 に一致|
|`value`|`[B, H, T, V]`、クエリと同じ dtype|
|`beta`|`[B, H, T]`、クエリと同じ dtype|
|`log_decay`|`[B, H, T]`、F32|
|`initial_state`|`[B, H, K, V]`、F32|
|`query_scale`| モデル構成に従って指定する有限の f32 |

すべてのテンソルは量子化されておらず、同じデバイス上に存在する必要があります。モデル固有の正規化後に Q/K を供給します。エントリ ポイントはそれを実行しません。この関数は、前述の `Runtime` および `RudaTensor` インポートを再利用します。

```rust
use rudnn::gated_delta::{
    GatedDeltaError, GatedDeltaInput, GatedDeltaOutput, chunk_gated_delta_rule,
};

fn delta_prefill<R: Runtime>(
    query: RudaTensor<R>,
    key: RudaTensor<R>,
    value: RudaTensor<R>,
    beta: RudaTensor<R>,
    log_decay: RudaTensor<R>,
    initial_state: RudaTensor<R>,
    query_scale: f32,
    chunk_size: usize,
) -> Result<GatedDeltaOutput<R>, GatedDeltaError> {
    chunk_gated_delta_rule(
        GatedDeltaInput {
            query, key, value, beta, log_decay, initial_state, query_scale,
        },
        chunk_size,
    )
}
```

返される `output` の形状は `[B, H, T, V]` で、dtype は query と同じです。`final_state` は F32 で、形状は `[B, H, K, V]` です。同じ系列の次の区間では、その `final_state` を `initial_state` として渡します。系列ごとに別の状態を保持してください。初期状態をインプレースで上書きすることはありません。

`chunk_size` は正でなければならず、三角形の作業領域 `4 × (chunk_size² + chunk_size)` バイトがデバイスのワークグループ当たりの共有メモリ上限を超えてはいけません。エントリポイントは最後のチャンクをパディングしますが、そのパディングは出力から除外します。無効な引数には `GatedDeltaError` を返します。

モデルレベルのテキストおよび画像の呼び出しについては、[ruLLM 推論ガイド](../../../docs/ja/model-inference.md) を参照してください。
