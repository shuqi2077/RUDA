# ruLLM

[English](../../README.md) | **简体中文** | [日本語](../ja/README.md) | [Deutsch](../de/README.md) | [Русский](../ru/README.md)

基于 Ruda 的 LLM 推理、模型加载与文本生成库。

ruLLM 将分词、模型加载、带缓存的自回归生成及请求调度与 Ruda 张量和计算库组合使用。

- Cargo package：`ruLLM`
- Rust crate：`rullm`

## 快速开始

配置 [NVIDIA 环境](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/getting-started.md)，并准备本地 [Qwen3.5-0.8B 模型目录](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/model-inference.md#准备本地模型)。

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo run --release --locked -p ruLLM --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "The capital of France is" 8 1
```

将 `./models/qwen35` 替换为自己的模型目录。最后两个参数分别为最大新生成 token 数和运行次数。未设置 `RUDA_CUDA_COMPILER` 时，示例默认使用直接 PTX；设为 `nvrtc` 可改用 CUDA C++ 编译路径。目标版本选择见 [PTX 配置](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/ptx.md)。

## 文档

- [模型加载与推理](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/model-inference.md)
- [Cargo features](../../Cargo.toml) · [模块入口](../../src/lib.rs)

## 模型加载与推理

[文档首页](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/README.md) · [训练](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/training.md) · [English](../../README.md)

ruLLM 提供模型加载、分词、带缓存的自回归生成和请求调度。Cargo package 名为 `ruLLM`，Rust 导入名为 `rullm`。

### 准备本地模型

将模型文件保存在同一目录：

| 文件 | 用途 |
| --- | --- |
| `config.json` | 模型结构和参数 |
| `model.safetensors` | 单文件权重 |
| `model.safetensors.index.json` 及其中列出的全部分片 | 分片权重，替代单文件权重 |
| `tokenizer.json` | 文本分词与解码 |
| `tokenizer_config.json` | Qwen2／Qwen2.5 pipeline 要求的 ChatML 模板 |
| `generation_config.json` | Qwen2／Qwen2.5 可选生成配置，读取 EOS 设置 |
| `preprocessor_config.json` | Qwen3.5 图片预处理配置，图片输入时需要 |

加载函数接收本地目录，不自动下载模型。按模型结构选择入口：

| 模型／输入 | 加载入口 |
| --- | --- |
| 非量化 Llama 文本 | `load_huggingface_llama_pipeline::<B>` |
| 非量化 Qwen2／Qwen2.5 文本 | `load_huggingface_qwen2_pipeline::<B>` |
| AWQ GEMM INT4 Qwen2／Qwen2.5 | `load_huggingface_awq_qwen2_pipeline::<R>` |
| 非量化 Qwen3.5 文本 | `load_huggingface_qwen35_text::<B>` |
| Qwen3.5 视觉编码器 | `load_huggingface_qwen35_vision::<B>` |
| Qwen3.5 图片与文本 | `load_huggingface_qwen35_multimodal::<B>` |

`B` 是张量 Backend，`R` 是设备 Runtime。Qwen3.5 文本入口使用 `qwen3_5_text` 配置及逐层 `layer_types`，不接收量化配置；不要把 AWQ 文件传给浮点加载入口。

### 运行文本生成示例

完成[环境配置](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/getting-started.md)后，在源码根目录运行。下面的相对路径分别指向两份本地模型目录：

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --locked -p ruLLM --features nvidia --example qwen2_generate -- ./models/qwen2 "Hello" 32 1
```

Qwen3.5 示例在未设置 `RUDA_CUDA_COMPILER` 时默认使用 Ruda IR → PTX。使用 `--release` 启用优化构建。直接 PTX 不支持的操作会报错，不回退到 NVRTC。PTX 版本需与目标 GPU 和驱动匹配，见 [PTX 后端参考](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/ptx.md)。

```powershell
Remove-Item Env:RUDA_CUDA_COMPILER -ErrorAction SilentlyContinue
$env:RUDA_PTX_VERSION = '8.0'
cargo run --release --locked -p ruLLM --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

保留 CUDA C++ / NVRTC 路径，可显式选择：

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --release --locked -p ruLLM --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

参数依次为模型目录、原始文本提示、最大新增 token 数、运行次数。Qwen2 示例后两个参数默认为 `32`、`1`；Qwen3.5 默认为 `8`、`1`，运行次数必须大于零。示例输出加载信息和生成结果的 JSON 行，生成行包含 `text`、`generated_token_ids`、`stopped_on_eos`。

这两个示例直接编码提示文本，不自动添加聊天模板。对话模型使用下面的 `render_chat` 调用准备提示。

### 在应用中生成对话回复

应用目录与 `RUDA` 源码目录同级时，依赖配置如下。

```toml
[dependencies]
rullm = { package = "ruLLM", path = "../RUDA/ruLLM", features = ["nvidia"] }
ruda-tensor-device = { path = "../RUDA/ruda-tensor-device", default-features = false, features = ["std", "cuda"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
half = "=2.7.1"

```

下面是完整的 `src/main.rs`，命令行接收模型目录和用户消息。后续函数示例复用这里的 `B`、导入及依赖：

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

`render_chat` 处理 Qwen2／Qwen2.5 的纯文本 ChatML 消息，第二个参数 `true` 添加 assistant 生成前缀。角色还可以是 `System`、`Assistant`；消息按会话顺序传入。该入口不接收工具调用消息。

`generate_text` 的两个布尔参数依次是 `add_special_tokens` 和 `skip_special_tokens`。已经渲染 ChatML 时不重复添加特殊 token；解码时可跳过特殊 token。`generated_text` 只含新增回复，`text` 包含提示及回复。

### 温度、top-k 与 top-p 采样

将渲染后的提示交给 `generate_text_sampled`：

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

| 参数 | 取值与效果 |
| --- | --- |
| `temperature` | 有限且大于 0；先缩放 logits |
| `top_k` | 再保留高分 token；0 关闭，边界同分 token 一并保留 |
| `top_p` | 最后按累计概率筛选；范围为 (0, 1]，1 关闭 |
| `seed` | 当前请求的采样随机种子；`None` 使用系统熵 |
| `max_new_tokens` | 只限制新增 token 数，不包含提示 |
| `eos_token_ids` | 生成任一指定 token 后停止 |

需要最大 logit 解码时使用 `generate_text`，而不是把温度设为零。固定 seed 固定的是采样随机序列，不代替模型及后端的数值一致性。

### 加载 AWQ 权重

AWQ 入口的泛型是 `CudaRuntime`，不是 `Cuda<bf16, i32>`：

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

返回的 pipeline 同样提供 `render_chat`、`encode`、`decode`、`generate_text` 和 `generate_text_sampled`。AWQ 配置要求 `quant_method = "awq"`、`bits = 4`、`version = "gemm"`、`zero_point = true`；`group_size` 为正数或 `-1`。线性投影使用 F16 激活，打包权重布局见 [ruBLAS INT4](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/rublas.md)。

### Qwen3.5 图片输入

`generate_image_files_greedy` 接收一条提示的 token ID、按提示顺序排列的 JPEG／PNG 路径、图片处理器和生成配置：

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

使用该模型的 `tokenizer.json` 编码 `prompt_ids`，并按模型提示模板为每张图片保留一个 `config.json` 中的 `image_token_id` 占位 token。图片路径数量和顺序必须与占位 token 一致。此入口内部完成占位展开，不要事先重复展开；它处理静态图片，不接收视频 token。

已经持有 RGB8 数据时，用 `Qwen35RgbImage { pixels, height, width }` 和 `generate_rgb_greedy`。像素为无行填充、交错存储的 RGB 字节。需要单独预处理时，`preprocess_files`／`preprocess_rgb` 返回 FP32 `patches`、二维 `shape` 和各图片的 `grids`。

同一模型处理多个请求时，将模型和处理器的加载移到请求循环外；生成函数的输出 `generated_token_ids` 可以交回相同 tokenizer 解码。

### 缓存与连续批处理

单请求生成入口自动创建并维护本次生成的缓存。手动驱动 Qwen3.5 时，调用 `model.new_cache()`，将提示张量传给 `forward_cached_last(tokens, &mut cache)`，之后只传新增 token；不要把已经缓存的整个前缀再次提交。

自行接入批量执行器时，使用 `ContinuousBatchScheduler`：

1. 用 `ContinuousBatchScheduler::new(config, kv_config)` 创建调度器。`ContinuousBatchConfig` 设置 `max_active_sequences` 和 `max_batch_tokens`；`PagedKvCacheConfig` 设置 `block_size`、`num_pages` 和 `max_sequence_length`，这些值都必须大于零。
2. 用 `submit(prompt_token_ids, generation)` 提交请求并保留返回的 `RequestId`。
3. 调用 `schedule()` 获取可执行的 `ScheduledBatch`。`kind` 区分 Prefill／Decode，`sequences` 给出每行的 token、起始位置、上下文长度和页表。
4. 执行器按行顺序运行模型、写入对应 KV 页，并选出每行的下一个 token。成功后调用 `complete_batch(batch.id, &generated_token_ids)`，每行必须对应一个结果 token。失败时调用 `fail_batch(batch.id)` 撤销该批的页预留。
5. 用 `pop_finished()` 取回完成请求及生成 token；然后继续调度。

调度器管理请求和页分配元数据；设备 KV 存储与模型执行由调用方的执行器负责。`token_matrix()`、`context_lengths()`、`flattened_block_table()` 可用于构造批量输入。存在未完成批次时，先完成或撤销该批次，再调用 `schedule()`。

接口参考：[ruLLM 导出](../../src/lib.rs)、[生成配置](../../src/generation.rs)、[调度器](../../src/continuous_batch.rs)。
