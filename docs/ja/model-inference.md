# モデルのロードと推論

[English](../en/model-inference.md) | [简体中文](../zh/model-inference.md) | **日本語** | [Deutsch](../de/model-inference.md) | [Русский](../ru/model-inference.md)

[ドキュメント](README.md) · [トレーニング](training.md) · [中文](../zh/model-inference.md)

ruLLM は、モデルの読み込み、トークン化、キャッシュされた自己回帰生成、およびリクエストのスケジューリングを提供します。 Cargo パッケージは `ruLLM` です。 Rust インポート名は `rullm` です。

## ローカルモデルの準備

モデル ファイルを 1 つのディレクトリに保存します。

|ファイル|目的|
| --- | --- |
|`config.json`|モデルのアーキテクチャとパラメータ|
|`model.safetensors`|単一ファイルの重み|
|`model.safetensors.index.json` およびリストされたすべてのシャード|単一のウェイト ファイルではなく、分割されたウェイト|
|`tokenizer.json`|テキストのトークン化とデコード|
|`tokenizer_config.json`|ChatML Qwen2/Qwen2.5 パイプラインに必要なテンプレート|
|`generation_config.json`|EOS 設定を提供するオプションの Qwen2/Qwen2.5 生成構成|
|`preprocessor_config.json`|Qwen3.5 画像前処理構成、画像入力に必要|

ローダーはローカル ディレクトリを受け入れ、モデルをダウンロードしません。アーキテクチャに応じたローダーを選択します。

|モデル/入力|ローダー|
| --- | --- |
|量子化されていないラマ テキスト|`load_huggingface_llama_pipeline::<B>`|
|量子化されていない Qwen2/Qwen2.5 テキスト|`load_huggingface_qwen2_pipeline::<B>`|
|AWQ GEMM INT4 Qwen2/Qwen2.5|`load_huggingface_awq_qwen2_pipeline::<R>`|
|量子化されていない Qwen3.5 テキスト|`load_huggingface_qwen35_text::<B>`|
|Qwen3.5 ビジョン エンコーダ|`load_huggingface_qwen35_vision::<B>`|
|Qwen3.5 の画像とテキスト|`load_huggingface_qwen35_multimodal::<B>`|

`B` はテンソル バックエンドです。 `R` はデバイスのランタイムです。 Qwen3.5 テキスト ローダーは、レイヤーごとの `layer_types` を持つ `qwen3_5_text` 構成を使用し、量子化構成を拒否します。 AWQ ファイルを浮動小数点ローダーに渡さないでください。

## テキスト生成サンプルの実行

[環境のセットアップ](getting-started.md) 後、ソース ルートからこれらのコマンドを実行します。相対パスは 2 つのローカル モデル ディレクトリを指します。

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --locked -p ruLLM --features nvidia --example qwen2_generate -- ./models/qwen2 "Hello" 32 1
```

Qwen3.5 の例では、`RUDA_CUDA_COMPILER` が設定されていない場合、デフォルトで Ruda IR → PTX になります。実行を最適化するには、`--release` を使用します。サポートされていない直接 PTX 操作は、NVRTC にフォールバックするのではなく、エラーを返します。 PTX のバージョンは、ターゲット GPU およびドライバーと一致する必要があります。 [PTX バックエンドリファレンス](ptx.md) を参照してください。

```powershell
Remove-Item Env:RUDA_CUDA_COMPILER -ErrorAction SilentlyContinue
$env:RUDA_PTX_VERSION = '8.0'
cargo run --release --locked -p ruLLM --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

CUDA C++ / NVRTC パスは、明示的に選択すると引き続き使用できます。

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --release --locked -p ruLLM --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

引数は、モデル ディレクトリ、生のテキスト プロンプト、新しいトークンの最大数、および実行回数です。最後の 2 つのデフォルトは、Qwen2 の場合は `32` と `1`、Qwen3.5 の場合は `8` と `1` です。実行カウントは正の値である必要があります。例では、ロードおよび生成のために JSON 行を出力します。生成ラインには、`text`、`generated_token_ids`、および `stopped_on_eos` が含まれます。

これらの例は、チャット テンプレートを追加せずに、プロンプトを直接エンコードします。会話型モデルの場合は、以下のように `render_chat` を使用してプロンプトを準備します。

## アプリケーションでチャット応答を生成する

`RUDA` ソース ディレクトリと並んでアプリケーション ディレクトリの場合は、これらの依存関係を使用します。

```toml
[dependencies]
rullm = { package = "ruLLM", path = "../RUDA/ruLLM", features = ["nvidia"] }
ruda-tensor-device = { path = "../RUDA/ruda-tensor-device", default-features = false, features = ["std", "cuda"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
half = "=2.7.1"

```

この完全な `src/main.rs` は、モデル ディレクトリとユーザー メッセージをコマンドライン引数として受け入れます。後の関数の例では、`B` エイリアス、インポート、依存関係を再利用します。

```rust
use half::bf16;
use ruda_tensor_device::cuda::{Cuda, CudaDevice};
use rullm::{
    GreedyGenerationConfig, Qwen2ChatMessage, Qwen2ChatRole,
    load_huggingface_qwen2_pipeline,
};
use std::{error::Error, io};

type B = Cuda<bf16, i32>;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let directory = args.next().ok_or_else(|| io::Error::other("model directory required"))?;
    let prompt = args.next().ok_or_else(|| io::Error::other("prompt required"))?;
    let device = CudaDevice::default();
    let pipeline = load_huggingface_qwen2_pipeline::<B>(&directory, &device)?;
    let messages = [Qwen2ChatMessage::new(Qwen2ChatRole::User, prompt)];
    let prompt = pipeline.render_chat(&messages, true)?;
    let output = pipeline.generate_text(
        &prompt,
        GreedyGenerationConfig {
            max_new_tokens: 32,
            eos_token_ids: pipeline.loaded.default_eos_token_ids.clone(),
        },
        false,
        true,
        &device,
    )?;
    println!("{}", output.generated_text);
    Ok(())
}
```

`render_chat` は、プレーンテキストの Qwen2/Qwen2.5 ChatML メッセージを処理します。 `true` を渡すと、アシスタント生成プレフィックスが追加されます。役割には、`System` および `Assistant` も含まれます。会話の順序に従ってメッセージを入力します。この API はツール呼び出しメッセージを受け入れません。

`generate_text` に対する 2 つのブール引数は、`add_special_tokens` と `skip_special_tokens` です。 ChatML のレンダリング後に特別なトークンを再度追加しないでください。デコードするとそれらをスキップできます。 `generated_text` には新しい応答のみが含まれますが、`text` にはプロンプトと応答の両方が含まれます。

## 温度、top-k、および top-p のサンプリング

レンダリングされたプロンプトを `generate_text_sampled` に渡します。

```rust
use rullm::{
    GeneratedText, HuggingFaceLoadError, HuggingFaceQwen2Pipeline,
    SamplingConfig, SamplingGenerationConfig,
};

fn sample_text(
    pipeline: &HuggingFaceQwen2Pipeline<B>,
    prompt: &str,
    device: &CudaDevice,
) -> Result<GeneratedText, HuggingFaceLoadError> {
    pipeline.generate_text_sampled(
        prompt,
        SamplingGenerationConfig {
            max_new_tokens: 32,
            eos_token_ids: pipeline.loaded.default_eos_token_ids.clone(),
            sampling: SamplingConfig {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.9,
                seed: Some(42),
            },
        },
        false,
        true,
        device,
    )
}
```

|パラメータ|値と効果|
| --- | --- |
|`temperature`|有限かつ正。 scales が最初にログオンします|
|`top_k`|高スコアのトークンを次に保持します。 0 はフィルタリングを無効にし、境界のタイは保持されます。|
|`top_p`|最後に累積確率によってフィルタリングします。範囲 (0, 1]、1 でフィルタリングが無効になります|
|`seed`|ローカル サンプリング シードを要求します。 `None` はシステム エントロピーを使用します|
|`max_new_tokens`|プロンプトを除き、新しいトークンを制限します|
|`eos_token_ids`|リストされたトークンを生成した後に停止します|

温度をゼロに設定するのではなく、最大ロジット デコードに `generate_text` を使用します。固定シードは、モデルまたはバックエンド間の数値の一貫性ではなく、サンプリングのランダム シーケンスを固定します。

## AWQ 重みの読み込み

AWQ ローダーは、汎用パラメーターとして `Cuda<bf16, i32>` ではなく `CudaRuntime` を受け取ります。

```rust
use ruda_driver_cuda::CudaRuntime;
use rullm::{HuggingFaceAwqQwen2Pipeline, load_huggingface_awq_qwen2_pipeline};

fn load_awq(
    directory: &str,
    device: &CudaDevice,
) -> Result<HuggingFaceAwqQwen2Pipeline<CudaRuntime>, HuggingFaceLoadError> {
    load_huggingface_awq_qwen2_pipeline::<CudaRuntime>(directory, device)
}
```

返されたパイプラインは、`render_chat`、`encode`、`decode`、`generate_text`、および `generate_text_sampled` も提供します。 AWQ 構成には、`quant_method = "awq"`、`bits = 4`、`version = "gemm"`、および `zero_point = true` が必要です。 `group_size` は正または `-1` でなければなりません。線形投影では、F16 アクティベーションを使用します。梱包重量レイアウトについては、[ruBLAS INT4](libraries/rublas.md) を参照してください。

## Qwen3.5画像入力

`generate_image_files_greedy` は、1 つのプロンプトのトークン ID、プロンプト順の JPEG/PNG パス、画像プロセッサー、および生成オプションを受け取ります。

```rust
use rullm::{
    Qwen35ImageProcessor, TokenGenerationOutput,
    load_huggingface_qwen35_multimodal,
};

fn generate_from_images(
    directory: &str,
    prompt_ids: &[i32],
    paths: &[&str],
    device: &CudaDevice,
) -> Result<TokenGenerationOutput, Box<dyn Error>> {
    let model = load_huggingface_qwen35_multimodal::<B>(directory, device)?;
    let processor = Qwen35ImageProcessor::from_huggingface(directory)?;
    let output = model.generate_image_files_greedy(
        prompt_ids,
        paths,
        &processor,
        &GreedyGenerationConfig {
            max_new_tokens: 32,
            eos_token_ids: vec![model.text.config.eos_token_id],
        },
    )?;
    Ok(output)
}
```

`prompt_ids` をモデルの `tokenizer.json` でエンコードします。モデルのプロンプト テンプレートに従い、`config.json` から `image_token_id` を使用して画像ごとに 1 つのプレースホルダー トークンを保持します。画像パスはプレースホルダーの数と順序と一致する必要があります。このエントリ ポイントは、プレースホルダーを内部で展開します。事前に展開しないでください。静止画像を処理し、ビデオ トークンを拒否します。

既存の RGB8 データの場合は、`Qwen35RgbImage { pixels, height, width }` を `generate_rgb_greedy` とともに使用します。ピクセルは行パディングなしでインターリーブされた RGB バイトです。個別の前処理の場合、`preprocess_files` および `preprocess_rgb` は、FP32 `patches`、2 次元の `shape`、およびイメージごとの `grids` を返します。

複数のリクエストを処理する場合は、リクエスト ループの外側でモデルとプロセッサをロードします。返された `generated_token_ids` を同じトークナイザーでデコードします。

## キャッシュと連続バッチ処理

単一リクエスト生成関数は、その世代のキャッシュを作成および維持します。 Qwen3.5 を手動で駆動するには、`model.new_cache()` を呼び出し、プロンプト テンソルを `forward_cached_last(tokens, &mut cache)` に渡し、その後、新しいトークンのみを渡します。キャッシュされたプレフィックス全体を再送信しないでください。

独自のバッチ エグゼキューターを統合するには、`ContinuousBatchScheduler` を使用します。

1. `ContinuousBatchScheduler::new(config, kv_config)` で作成します。 `ContinuousBatchConfig` は `max_active_sequences` と `max_batch_tokens` を設定します。 `PagedKvCacheConfig` は、`block_size`、`num_pages`、および `max_sequence_length` を設定します。すべてがポジティブでなければなりません。
2. `submit(prompt_token_ids, generation)` を使用してリクエストを送信し、返されたそれぞれの `RequestId` を保持します。
3. `ScheduledBatch` を取得するには、`schedule()` を呼び出します。 `kind` はプレフィル/デコードを区別します。 `sequences` は、各行のトークン、開始位置、コンテキストの長さ、およびブロック テーブルを提供します。
4. モデルを行順に実行し、対応する KV ページを書き込み、各行の次のトークンを選択します。成功したら、行ごとに 1 つの結果トークンを指定して `complete_batch(batch.id, &generated_token_ids)` を呼び出します。失敗した場合は、`fail_batch(batch.id)` を呼び出して、そのバッチのページ予約をキャンセルします。
5. `pop_finished()` を使用して、完了したリクエストと生成されたトークンを取得し、スケジューリングを続行します。

スケジューラはリクエストとページ割り当てメタデータを管理します。呼び出し元のエグゼキュータは、デバイス KV ストレージとモデルの実行を所有しています。 `token_matrix()`、`context_lengths()`、および `flattened_block_table()` を使用してバッチ入力を構築します。 `schedule()` を再度呼び出す前に、未処理のバッチを完了するかキャンセルしてください。

API 参照: [ruLLM エクスポート](../../ruLLM/src/lib.rs)、[生成オプション](../../ruLLM/src/generation.rs)、[スケジューラー](../../ruLLM/src/continuous_batch.rs)。
