# 示例与教程

[文档首页](README.md) · [快速开始](getting-started.md) · [English](../en/samples.md) | [日本語](../ja/samples.md) | [Deutsch](../de/samples.md) | [Русский](../ru/samples.md)

## 1. NVIDIA 运行时示例

入口：[ptx-runtime](../../ruda-driver-cuda/examples/ptx_runtime.rs)。

基础用例对长度 1、63、64、65、257 各执行两次 FP32 加法，逐元素检查结果并检查输出尾部 16 个哨兵值。它演示设备选择、数据上传、参数绑定、尾部边界、回读及重复执行。

构建和后端选择见[快速开始](getting-started.md)。

## 2. 定向用例

以下参数放在命令的 `--` 后。例如运行张量用例：

```powershell
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --tensor
```

| 参数 | 内容 | 示例源码 |
| --- | --- | --- |
| `--tensor` | 张量元数据与布局相关检查 | [tensor.rs](../../ruda-driver-cuda/examples/ptx_runtime/tensor.rs) |
| `--shared` | 共享内存相关检查 | [shared.rs](../../ruda-driver-cuda/examples/ptx_runtime/shared.rs) |
| `--half` | FP16／BF16 相关检查 | [half_precision.rs](../../ruda-driver-cuda/examples/ptx_runtime/half_precision.rs) |
| `--bitwise` | 位运算定向检查 | [bitwise.rs](../../ruda-driver-cuda/examples/ptx_runtime/bitwise.rs) |
| `--shared-over-limit` | 共享内存超限诊断 | [shared.rs](../../ruda-driver-cuda/examples/ptx_runtime/shared.rs) |
| `--expect-cold` | 断言发生编译且没有磁盘缓存命中 | [主入口](../../ruda-driver-cuda/examples/ptx_runtime.rs) |
| `--expect-warm` | 断言不重新编译且实际命中磁盘缓存 | [主入口](../../ruda-driver-cuda/examples/ptx_runtime.rs) |

`--shared-over-limit` 是提前返回的独立分支，不应与冷／热缓存检查合并使用。

## 3. 冷缓存与热缓存

每条编译路径使用独立的新缓存目录，同一路径的冷／热两次运行复用该目录。

按[快速开始](getting-started.md)选择编译路径后，在同一 PowerShell 会话执行：

```powershell
$env:RUDA_PTX_TEST_CACHE = 'target/ptx-example-cache-' + [guid]::NewGuid().ToString('N')
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --expect-cold
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --expect-warm
```

两次命令之间不要改变编译器、输入参数或缓存路径。

## 4. 计算库示例

运行示例前，先准备[构建环境](getting-started.md)。

| 任务 | 命令／指南 | 输出检查 |
| --- | --- | --- |
| FP32 CSR 矩阵向量乘 | `cargo run --locked -p ruSPARSE --features cuda --example csrmv` | 示例回读并校验 `[7.0, 2.0, 18.5]` |
| CUDA Ring AllReduce | `cargo run --locked -p ruCCL --features cuda --example all_reduce` | GPU 0 上四个逻辑 rank，257 个元素，Sum／Mean 与输入保持 |

## 5. 训练与模型推理

- [训练与状态保存](training.md)：前向、反向、梯度累积及训练记录。
- [模型加载与推理](model-inference.md)：Qwen2／Qwen3.5 示例命令、对话、采样、AWQ 与图片输入。
