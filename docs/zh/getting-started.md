# 安装与快速开始

[文档首页](README.md) · [下一步：编程指南](programming-guide.md) · [English](../en/getting-started.md) | [日本語](../ja/getting-started.md) | [Deutsch](../de/getting-started.md) | [Русский](../ru/getting-started.md)

## 1. 选择使用层级

| 任务 | 入口 |
| --- | --- |
| 编写 GPU Kernel | `ruda-kernel::dsl` 与设备运行时 |
| 使用矩阵乘、FFT、归约等运算 | [计算库](libraries/README.md) |
| 使用张量及框架 | [张量与框架指南](tensor-framework.md) |
| 训练模型与保存状态 | [训练指南](training.md) |
| 加载模型并生成文本或处理图片 | [模型推理指南](model-inference.md) |
| 接入设备后端 | [Driver API](driver-api.md) |

当前以源码 workspace 为入口。

## 2. 准备 NVIDIA 环境

需要 Rust／Cargo、目标平台的链接工具、NVIDIA GPU 驱动及 CUDA Toolkit。现有 CUDA 后端仍包含 NVRTC 和 CUDA 工具包依赖，启用直接 PTX 不会自动移除这些构建依赖。

在源码根目录检查环境：

```powershell
rustc --version --verbose
cargo --version
nvidia-smi
nvcc --version
cargo metadata --no-deps --format-version 1 --offline --locked
```

这些命令不编译 Ruda。`--offline` 要求解析所需的依赖已在本机缓存。

`CUDA_PATH` 可指定 CUDA Toolkit 根目录。Windows 上应指向实际安装的版本目录，而不是只有多个版本子目录的父目录。定位实现见 [CUDA 安装路径接口](../../ruda-driver-cuda/src/lib.rs)。

## 3. 构建已有示例

以下命令用于具备环境后的源码构建。

```powershell
cargo build --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

`ptx-runtime` 示例要求启用 `direct-ptx` feature；启用该 feature 本身不会切换默认编译器。

## 4. 选择编译路径并运行

在单独的 PowerShell 会话中选择一条路径。每条命令失败时先处理错误，再继续。

默认 CUDA C++／NVRTC 路径：

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

直接 PTX 路径：

```powershell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

PTX 版本需与目标 GPU 和驱动匹配，见 [PTX 后端参考](ptx.md)。

示例对多个长度执行 FP32 加法，检查输出、尾部哨兵及重复执行，并打印缓存计数。具体检查与可选用例见[示例与教程](samples.md)。

## 5. 继续开发

先从示例中的设备选择、数据上传和 Kernel 启动理解执行流程，再阅读[编程指南](programming-guide.md)。出现构建、驱动加载或运行错误时，按[调试与诊断](debugging.md)定位所在阶段。
