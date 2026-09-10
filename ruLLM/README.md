# ruLLM

**English** | [简体中文](https://github.com/shuqi2077/RUDA/blob/main/ruLLM/docs/zh/README.md) | [日本語](https://github.com/shuqi2077/RUDA/blob/main/ruLLM/docs/ja/README.md) | [Deutsch](https://github.com/shuqi2077/RUDA/blob/main/ruLLM/docs/de/README.md) | [Русский](https://github.com/shuqi2077/RUDA/blob/main/ruLLM/docs/ru/README.md)

LLM inference, model loading, and text generation on Ruda.

ruLLM combines tokenization, model loading, cached autoregressive generation, and request scheduling with Ruda tensors and compute libraries.

- Cargo package: `ruda-llm`
- Rust crate: `rullm`

## Multi-backend and execution reliability candidate

The second-round candidate adds HIP feature wiring, capability-gated packed
kernels, checked memory/shape planning, bounded replica workers, completion-fence
contracts, recovery decisions and explicit performance fallback. It does not
implement model sharding or custom accelerator kernels.

## Generation and scheduler extensions

This source revision adds token callbacks, token-sequence stopping, cooperative
cancellation, bounded scheduler queuing, optional full-sequence KV admission,
and atomic multi-reservation cache operations. Existing generation configuration
structs and entry points remain available.

Sampling now reuses scratch buffers and selects the top-k boundary without a
full vocabulary sort.

```sh
cargo test --locked -p ruda-llm --lib --test generation_control
cargo run --release --locked -p ruda-llm --example sampling_bench -- 32000 100
cargo run --release --locked -p ruda-llm --features nvidia --example qwen2_stream -- ./models/qwen2 "Hello" 32
```

## Quick Start

Set up the [NVIDIA environment](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md) and prepare a local [Qwen3.5-0.8B model directory](https://github.com/shuqi2077/RUDA/blob/main/docs/en/model-inference.md#prepare-a-local-model).

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "The capital of France is" 8 1
```

Replace `./models/qwen35` with your model directory. The final arguments select the maximum new tokens and number of runs. The example defaults to direct PTX when `RUDA_CUDA_COMPILER` is unset. Set it to `nvrtc` to use the CUDA C++ compilation path instead. See [PTX configuration](https://github.com/shuqi2077/RUDA/blob/main/docs/en/ptx.md) for target-version selection.

## Documentation

- [Model loading and inference](https://github.com/shuqi2077/RUDA/blob/main/docs/en/model-inference.md)
- [Cargo features](https://github.com/shuqi2077/RUDA/blob/main/ruLLM/Cargo.toml) · [Module exports](https://github.com/shuqi2077/RUDA/blob/main/ruLLM/src/lib.rs)

## Model Loading and Inference

[Documentation](https://github.com/shuqi2077/RUDA/blob/main/docs/en/README.md) · [Training](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md) · [中文](https://github.com/shuqi2077/RUDA/blob/main/ruLLM/docs/zh/README.md)

ruLLM provides model loading, tokenization, cached autoregressive generation, and request scheduling. Its Cargo package is `ruda-llm`; its Rust import name is `rullm`.

### Prepare a local model

Keep the model files in one directory:

| File | Purpose |
| --- | --- |
| `config.json` | Model architecture and parameters |
| `model.safetensors` | Single-file weights |
| `model.safetensors.index.json` and every listed shard | Sharded weights, instead of a single weight file |
| `tokenizer.json` | Text tokenization and decoding |
| `tokenizer_config.json` | ChatML template required by Qwen2/Qwen2.5 pipelines |
| `generation_config.json` | Optional Qwen2/Qwen2.5 generation configuration supplying EOS settings |
| `preprocessor_config.json` | Qwen3.5 image preprocessing configuration, required for image input |

Loaders accept local directories and do not download models. Select the loader for your architecture:

| Model/input | Loader |
| --- | --- |
| Unquantized Llama text | `load_huggingface_llama_pipeline::<B>` |
| Unquantized Qwen2/Qwen2.5 text | `load_huggingface_qwen2_pipeline::<B>` |
| AWQ GEMM INT4 Qwen2/Qwen2.5 | `load_huggingface_awq_qwen2_pipeline::<R>` |
| Unquantized Qwen3.5 text | `load_huggingface_qwen35_text::<B>` |
| Qwen3.5 vision encoder | `load_huggingface_qwen35_vision::<B>` |
| Qwen3.5 images and text | `load_huggingface_qwen35_multimodal::<B>` |

`B` is a tensor Backend; `R` is a device Runtime. The Qwen3.5 text loader uses a `qwen3_5_text` configuration with per-layer `layer_types` and rejects quantization configurations. Do not pass AWQ files to a floating-point loader.

### Run the text generation examples

After [setting up the environment](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md), run these commands from the source root. The relative paths point to two local model directories:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --locked -p ruda-llm --features nvidia --example qwen2_generate -- ./models/qwen2 "Hello" 32 1
```

The Qwen3.5 example defaults to Ruda IR → PTX when `RUDA_CUDA_COMPILER` is unset. Use `--release` for optimized execution. Unsupported direct PTX operations return errors rather than falling back to NVRTC. The PTX version must match the target GPU and driver; see the [PTX backend reference](https://github.com/shuqi2077/RUDA/blob/main/docs/en/ptx.md).

```powershell
Remove-Item Env:RUDA_CUDA_COMPILER -ErrorAction SilentlyContinue
$env:RUDA_PTX_VERSION = '8.0'
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

The CUDA C++ / NVRTC path remains available with an explicit selection:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

Arguments are the model directory, raw text prompt, maximum new tokens, and run count. The last two default to `32` and `1` for Qwen2, or `8` and `1` for Qwen3.5. Run count must be positive. Examples print JSON lines for loading and generation. Generation lines include `text`, `generated_token_ids`, and `stopped_on_eos`.

These examples encode the prompt directly without adding a chat template. For conversational models, prepare the prompt with `render_chat` as below.

### Generate a chat reply in an application

For an application directory alongside the `RUDA` source directory, use these dependencies.

```toml
[dependencies]
rullm = { package = "ruda-llm", path = "../RUDA/ruLLM", features = ["nvidia"] }
ruda-tensor-device = { path = "../RUDA/ruda-tensor-device", default-features = false, features = ["std", "cuda"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
half = "=2.7.1"

```

This complete `src/main.rs` accepts a model directory and user message as command-line arguments. Later function examples reuse its `B` alias, imports, and dependencies:

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

`render_chat` handles plain-text Qwen2/Qwen2.5 ChatML messages. Passing `true` adds the assistant generation prefix. Roles also include `System` and `Assistant`; supply messages in conversation order. This API does not accept tool-call messages.

The two boolean arguments to `generate_text` are `add_special_tokens` and `skip_special_tokens`. Do not add special tokens again after rendering ChatML; decoding can skip them. `generated_text` contains only the new reply, while `text` includes both prompt and reply.

### Temperature, top-k, and top-p sampling

Pass the rendered prompt to `generate_text_sampled`:

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

| Parameter | Values and effect |
| --- | --- |
| `temperature` | Finite and positive; scales logits first |
| `top_k` | Keeps high-scoring tokens next; 0 disables filtering, ties at the boundary are retained |
| `top_p` | Finally filters by cumulative probability; range (0, 1], with 1 disabling filtering |
| `seed` | Request-local sampling seed; `None` uses system entropy |
| `max_new_tokens` | Limits new tokens, excluding the prompt |
| `eos_token_ids` | Stops after generating any listed token |

Use `generate_text` for maximum-logit decoding rather than setting temperature to zero. A fixed seed fixes the sampling random sequence, not numerical consistency across models or backends.

### Load AWQ weights

The AWQ loader takes `CudaRuntime`, not `Cuda<bf16, i32>`, as its generic parameter:

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

The returned pipeline also provides `render_chat`, `encode`, `decode`, `generate_text`, and `generate_text_sampled`. AWQ configuration requires `quant_method = "awq"`, `bits = 4`, `version = "gemm"`, and `zero_point = true`. `group_size` must be positive or `-1`. Linear projections use F16 activations. See [ruBLAS INT4](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rublas.md) for packed weight layouts.

### Qwen3.5 image input

`generate_image_files_greedy` takes token IDs for one prompt, JPEG/PNG paths in prompt order, an image processor, and generation options:

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

Encode `prompt_ids` with the model's `tokenizer.json`. Follow the model's prompt template, keeping one placeholder token per image using `image_token_id` from `config.json`. Image paths must match placeholder count and order. This entry point expands placeholders internally; do not expand them beforehand. It handles still images and rejects video tokens.

For existing RGB8 data, use `Qwen35RgbImage { pixels, height, width }` with `generate_rgb_greedy`. Pixels are interleaved RGB bytes without row padding. For separate preprocessing, `preprocess_files` and `preprocess_rgb` return FP32 `patches`, a two-dimensional `shape`, and per-image `grids`.

When serving multiple requests, load the model and processor outside the request loop. Decode the returned `generated_token_ids` with the same tokenizer.

### Caches and continuous batching

Single-request generation functions create and maintain a cache for that generation. To drive Qwen3.5 manually, call `model.new_cache()`, pass the prompt tensor to `forward_cached_last(tokens, &mut cache)`, then pass only new tokens. Do not resubmit the entire cached prefix.

To integrate your own batch executor, use `ContinuousBatchScheduler`:

1. Create it with `ContinuousBatchScheduler::new(config, kv_config)`. `ContinuousBatchConfig` sets `max_active_sequences` and `max_batch_tokens`. `PagedKvCacheConfig` sets `block_size`, `num_pages`, and `max_sequence_length`. All must be positive.
2. Submit requests with `submit(prompt_token_ids, generation)` and retain each returned `RequestId`.
3. Call `schedule()` for a `ScheduledBatch`. Its `kind` distinguishes Prefill/Decode; `sequences` supplies each row's tokens, start position, context length, and block table.
4. Run the model in row order, write the corresponding KV pages, and select each row's next token. On success, call `complete_batch(batch.id, &generated_token_ids)` with one result token per row. On failure, call `fail_batch(batch.id)` to cancel that batch's page reservations.
5. Retrieve completed requests and generated tokens with `pop_finished()`, then continue scheduling.

The scheduler manages requests and page-allocation metadata. The caller's executor owns device KV storage and model execution. Use `token_matrix()`, `context_lengths()`, and `flattened_block_table()` to construct batch inputs. Complete or cancel an outstanding batch before calling `schedule()` again.

API reference: [ruLLM exports](https://github.com/shuqi2077/RUDA/blob/main/ruLLM/src/lib.rs), [Generation options](https://github.com/shuqi2077/RUDA/blob/main/ruLLM/src/generation.rs), [Scheduler](https://github.com/shuqi2077/RUDA/blob/main/ruLLM/src/continuous_batch.rs).
