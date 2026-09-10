# ruLLM

[English](../../README.md) | **简体中文**

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
