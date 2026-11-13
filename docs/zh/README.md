# Ruda 文档

[文档目录](../README.md) · [English](../en/README.md) | [日本語](../ja/README.md) | [Deutsch](../de/README.md) | [Русский](../ru/README.md)

从第一个 GPU Kernel，到领域计算库、张量训练和本地模型推理。按任务开始，再查阅概念指南与 API 参考。

本手册对应 `v0.3.0` 发布系列。Rust 子库继续独立版本管理，源码版本以各自 Cargo 清单为准，已发布的 rustdoc 快照见 [Rust API 参考](rust-api.md)。请安装匹配的 `ruda-torch==0.3.0` Release wheel，或构建对应源码版本；之后仅存在于 `main` 的 API 需要匹配的源码构建。

## 从这里开始

| 我想做什么 | 阅读路径 |
| --- | --- |
| 运行第一个 GPU Kernel | [安装与快速开始](getting-started.md) → [编程指南](programming-guide.md) |
| 在 `ruda:0` 上使用 PyTorch | [原生 PyTorch 快速上手](native-pytorch-quickstart.md) → [原生 API](native-pytorch-api.md) |
| 安装 v0.3.0 wheel 或使用压缩量化 | [Release 安装](../../ruda-torch/README.md#release-wheels) → [GPTQ／GGUF 与量化 KV](native-pytorch-api.md#packed-量化与权重导入) |
| 使用矩阵、稀疏或神经网络算子 | [计算库](libraries/README.md) → [张量与框架](tensor-framework.md) |
| 训练模型、累积梯度与保存状态 | [训练与状态保存](training.md) |
| 使用 LoRA 或 NF4 微调本地模型 | [微调与恢复](finetuning.md) |
| 加载本地模型、生成文本或处理图片 | [模型加载与推理](model-inference.md) |

## 入门

- [安装与快速开始](getting-started.md)：源码环境、后端选择、第一个示例。
- [原生 PyTorch 快速上手](native-pytorch-quickstart.md)：安装、张量传输、一步 FP32 训练与问题定位。
- [示例与教程](samples.md)：向量加法、张量、共享内存及半精度用例。

## 编程指南

- [FlashInfer 原生迁移](flashinfer.md)：Rust 接口、数据格式与设备缓冲区约定（不包含于 v0.3.0 wheel）。

- [Ruda 编程指南](programming-guide.md)：主机与设备、执行层级、内存、同步及安全边界。
- [张量与框架指南](tensor-framework.md)：设备张量、领域库分发、融合与自动微分。
- [张量实用示例](tensor-recipes.md)：矩阵乘、布局、dtype 选择与梯度的完整 CPU 示例。
- [后端选择与组合](backend-composition.md)：CUDA／ROCm／WGPU 选择、本地路由与远程执行。
- [数据管线与模型存储](data-and-storage.md)：样本组批、权重保存与检查点格式导入。
- [训练与状态保存](training.md)：训练步、FP32 主参数／累积、混合存储 checkpoint、token 加权副本、可微分集合通信和学习率调度。
- [Rank、设备与分布式训练](distributed-training.md)：显式 rendezvous、设备映射、集合调用顺序、加权梯度与逐 rank 恢复。
- [Python 并行训练、数据与恢复](parallel-and-data.md)：TP／FSDP／ZeRO、流水线、sampler、checkpoint 格式与完整更新调优。
- [架构组件与 Python Muon](architecture-training.md)：mHC、DSA／CSA／HCA、压缩 KV 缓存与混合模型组合。
- [LoRA 与 NF4 微调](finetuning.md)：准确目标选择、流式权重、因果监督、token 加权累积、适配器与断点恢复。
- [通用 PyTorch 模型编译](model-compiler.md)：AOT 前向／反向、原生分段、配置及缓存所有权。
- [固定地址 PyTorch 子图](static-pytorch-graphs.md)：显式 GraphOp、输出生命周期、工作区复用与一阶训练。
- [集合通信与训练接入](distributed-guide.md)：peer 注册、显式 rank 所有权、张量适配与梯度同步。
- [模型加载与推理](model-inference.md)：ruLLM、文本与图片输入、采样、AWQ 和连续批处理。

## 编译与底层执行

- [编译器指南](compiler-guide.md)：Rust Kernel 前端、IR、CUDA C++／NVRTC 和直接 PTX。
- [昇腾编译器与 CANN 执行](ascend-guide.md)：Map／行程序、构建物生成、绑定布局与 session 生命周期。
- [PTX 后端参考](ptx.md)：目标配置、编译产物、支持边界与错误行为。
- [全栈自动调优](stack-autotuning.md)：已接入算子、离线校准、计时、缓存身份与策略参数。

## API 参考

- [Rust API 参考](rust-api.md)：运行时、Kernel、计算库、张量与模型 crate 的版本化 rustdoc。
- [Python 公开 API 签名](python-api-signatures.md)：公开包／模块函数、构造器、方法、字段与准确源码位置。
- [全栈 Crate 与 API 索引](api-reference.md)：整个 workspace 的包名、Rust 导入名、职责与源码入口。
- [Runtime API](runtime-api.md)：设备客户端、内存、提交、回读与同步。
- [Driver API 与后端](driver-api.md)：后端类型、设备选择和运行时接入。
- [原生 PyTorch API](native-pytorch-api.md)：设备／组件版本、BLAS／linalg／FFT 与显式 plan、shape／dtype 契约、归一化、优化器、流、注意力、序列训练与量化。
- [计算库参考](libraries/README.md)：按领域选择库、Cargo feature 与接口入口。

## 计算库

| 库 | 手册 |
| --- | --- |
| ruBLAS | [线性代数与分组矩阵乘](libraries/rublas.md) |
| ruDNN | [神经网络算子与 MoE](libraries/rudnn.md) |
| ruTENSOR | [张量收缩、einsum 与置换](libraries/rutensor.md) |
| ruPRIM | [归约、扫描与索引](libraries/ruprim.md) |
| ruFFT | [快速傅里叶变换](libraries/rufft.md) |
| ruRAND | [随机数生成](libraries/rurand.md) |
| ruSPARSE | [稀疏计算](libraries/rusparse.md) |
| ruCCL | [集合通信](libraries/ruccl.md) |
| ruSOLVER | [线性方程与矩阵分解](libraries/rusolver.md) |
| ruINTEGRATE | [数值积分与初值 ODE](libraries/ruintegrate.md) |

## 调试与兼容性

- [调试与诊断](debugging.md)：编译错误、异步错误、缓存与数值检查。
- [兼容性指南](compatibility.md)：CUDA 开发概念对照、后端差异、API 与编译路径边界。
- [贡献指南](CONTRIBUTING.md)：问题反馈与开发约定。

首次使用可按“快速开始 → 编程指南 → 所需计算库”阅读；开发后端或内核时，再查阅编译器与 API 参考。
