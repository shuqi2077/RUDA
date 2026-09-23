# Ruda — Rust 高性能计算库

[English](../../README.md) | **简体中文** | [日本語](../ja/project.md) | [Deutsch](../de/project.md) | [Русский](../ru/project.md)

Ruda 是 Rust 高性能计算库，正在构建从 GPU Kernel、编译器、运行时到数学计算、张量与模型的完整软件栈。

Ruda 面向 PTX、HIP 及自有 ISA 构建 Rust 编译与执行路径，同时保留 CUDA C++ 编译路径。通过底层受控的 `unsafe` 封装与上层 Rust 类型系统、所有权及借用机制，兼顾底层性能控制与上层内存安全。

## 快速开始

需要 Git、Rust/Cargo、链接工具链、NVIDIA GPU 及驱动和 CUDA Toolkit。安装详情见[环境配置](getting-started.md)。

### 使用已发布的 crate

在应用的 `Cargo.toml` 中添加 [CUDA 后端](https://crates.io/crates/ruda-driver-cuda)：

```toml
[dependencies]
ruda-driver-cuda = { version = "0.1", features = ["direct-ptx"] }
```

以下示例从源码目录运行。

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
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "The capital of France is" 8 1
```

示例打印生成文本及 token ID。若要改用 CUDA C++ / NVRTC 路径，在运行任一示例前将 `RUDA_CUDA_COMPILER` 设为 `nvrtc`。

### 使用原生 PyTorch 后端

`ruda-torch` 在单张 NVIDIA GPU 上注册 PyTorch 设备 `ruda:0`。安装 PyTorch、setuptools 并准备 C++20 编译器后，沿用上面的 PTX 环境设置，在仓库根目录执行以下命令。Windows 使用 x64 MSVC 开发者终端。

```sh
cargo build --locked -p ruda-torch-native
python -m pip install --no-build-isolation --no-deps -e ./ruda-torch/python
```

默认加载器会自动找到上述 debug 构建。使用 release 构建或其他位置的动态库时，将 `RUDA_TORCH_LIBRARY` 设为其路径。Rust 动态库与 C++ 扩展必须同时使用 **ABI 9**，升级时一起重建。

```python
import torch
import ruda_torch

x = torch.arange(4, dtype=torch.float32).to("ruda:0")
print((x + x).cpu())
```

预构建 Windows wheel 位于 [RUDA Torch Windows build](https://github.com/shuqi2077/RUDA/actions/workflows/ruda-torch-windows.yml) 成功运行的产物中。下载 wheel 产物并解压，用 `python -m pip install --no-deps` 安装其中的 `.whl` 文件。wheel 包含原生 DLL，面向 Windows x64、CPython 3.13 和 PyTorch `2.13.0+cu130`，需先安装匹配的 PyTorch。产物保留七天。`ruda-torch-native` 由源码构建，不作为 crates.io 包提供。

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
| PyTorch 接入 | `ruda-torch-native`（Rust）、`ruda_torch`（Python） |
| 模型与数据 | `ruda-model`、`ruda-nn`、`ruda-optim`、`ruda-store`、`ruda-dataset` |

## 原生 GPU 推理

- **算子**：原生 PyTorch 矩阵运算接入 ruBLAS；保留 FP16/BF16 存储的计算路径、末轴融合 LayerNorm/RMSNorm，以及线程束并行 Softmax/归约，减少中间张量和独立内核提交。
- **分页 GQA 与 MLA**：ruDNN 公共内核直接读取物理 KV 页，处理变长 prefill/decode。`ruda_torch.PagedAttentionPlan` 支持 `splits=1..32`、FP32 分段结果合并及工作区复用，默认 `splits=1`；共享缓存写入仍保留写时复制保护。
- **MoE**：分组 sigmoid 路由与分段专家矩阵乘直接使用设备端专家偏移。FP16/BF16 Tensor Core 路径需显式选择，现有专家入口默认保留标量 GPU 策略。
- **流与事件**：`ruda_torch.Stream`、`Event` 和 `record_stream` 接入原生运行时。默认同步提交；在首次原生提交前设置 `RUDA_TORCH_ASYNC=1` 可启用异步提交。显式同步和主机回读仍等待完成。

分页注意力要求连续、同精度、同设备和同执行队列的 FP32/FP16/BF16 张量，仅支持前向，不支持任意外部掩码或量化 KV 缓存。MLA/MoE 是可复用组件，完整模型适配仍需提供投影、位置编码、路由参数和缓存所有权管理。

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
