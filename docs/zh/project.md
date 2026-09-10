# Ruda — Rust 高性能计算库

[English](../../README.md) | **简体中文**

Ruda 是 Rust 高性能计算库，正在构建从 GPU Kernel、编译器、运行时到数学计算、张量与模型的完整软件栈。

Ruda 面向 PTX、HIP 及自有 ISA 构建 Rust 编译与执行路径，同时保留 CUDA C++ 编译路径。通过底层受控的 `unsafe` 封装与上层 Rust 类型系统、所有权及借用机制，兼顾底层性能控制与上层内存安全。

## 快速开始

需要 Git、Rust/Cargo、链接工具链、NVIDIA GPU 及驱动和 CUDA Toolkit。安装详情见[环境配置](getting-started.md)。

### 克隆源码

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
```

### 运行 GPU Kernel

在终端中选择直接 PTX 编译器：

```sh
# Bash
export RUDA_CUDA_COMPILER=ptx
export RUDA_PTX_VERSION=8.0
```

```powershell
# PowerShell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
```

然后构建并运行示例：

```sh
cargo run --release --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

示例在 GPU 上执行 FP32 加法，打印 `PASS` 行及编译缓存计数。请选择 GPU 和驱动支持的 [PTX 版本](ptx.md)。

### 使用 ruLLM 生成文本

将本地 Qwen3.5-0.8B 模型放在 `./models/qwen35`，或将以下路径替换为自己的模型目录。仓库不包含模型文件，准备方式见[模型配置](model-inference.md#准备本地模型)。

```sh
cargo run --release --locked -p ruLLM --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "The capital of France is" 8 1
```

示例打印生成文本及 token ID。若要改用 CUDA C++ / NVRTC 路径，在运行任一示例前将 `RUDA_CUDA_COMPILER` 设为 `nvrtc`。

## 软件栈组织

一个仓库，多个职责清晰的 crate。从领域计算库到上层框架，按层组织、协同开发。

| 层 | 组件 |
| --- | --- |
| 公共契约 | `ruda-core` |
| 编译与 Kernel | `ruda-compiler`、`ruda-kernel`、宏组件 |
| 运行时与驱动后端 | `ruda`、`ruda-driver-cuda/cpu/wgpu/hip` |
| 领域计算库 | ruBLAS、ruDNN、ruPRIM、ruFFT、ruRAND、ruSPARSE |
| 集合通信 | ruCCL、`ruda-communication` |
| 张量与框架 | `ruda-tensor*`、`ruda-autodiff`、`ruda-fusion` |
| 模型与数据 | `ruda-model`、`ruda-nn`、`ruda-optim`、`ruda-store`、`ruda-dataset` |

## 通向硬件

- **NVIDIA GPU**：默认使用 CUDA C++ → NVRTC → PTX 编译路径，同时提供显式选择的 IR → PTX 直接生成路径，均通过 NVIDIA 驱动执行。
- **更多执行后端**：CPU、WGPU 与 HIP 已有后端源码，具体支持范围见[兼容性说明](compatibility.md)。

## 探索与参与

- [Ruda 文档](README.md)：快速开始、编程指南、编译器、API 参考与计算库手册。
- [NVIDIA 演示](getting-started.md)：了解示例与运行条件。
- [贡献指南](CONTRIBUTING.md)：参与算子、编译器、运行时和框架开发。

如果你关心 Rust、GPU Kernel、编译器或高性能计算，欢迎一起把这套软件栈做深、做快。

## 来源与许可

[第三方许可声明](../../THIRD_PARTY_NOTICES.md)

项目自身有权授权的 Ruda 原创软件代码采用 [Apache License 2.0](../../LICENSE)。第三方文件继续适用其原有许可；迁入组件中的 `MIT OR Apache-2.0` 声明不会被根许可证覆盖。
