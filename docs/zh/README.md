# Ruda 文档

[文档目录](../README.md) · [English](../en/README.md) | [日本語](../ja/README.md) | [Deutsch](../de/README.md) | [Русский](../ru/README.md)

从第一个 GPU Kernel，到领域计算库、张量训练和本地模型推理。按任务开始，再查阅概念指南与 API 参考。

## 从这里开始

| 我想做什么 | 阅读路径 |
| --- | --- |
| 运行第一个 GPU Kernel | [安装与快速开始](getting-started.md) → [编程指南](programming-guide.md) |
| 使用矩阵、稀疏或神经网络算子 | [计算库](libraries/README.md) → [张量与框架](tensor-framework.md) |
| 训练模型、累积梯度与保存状态 | [训练与状态保存](training.md) |
| 使用 LoRA 或 NF4 微调本地模型 | [微调与恢复](finetuning.md) |
| 加载本地模型、生成文本或处理图片 | [模型加载与推理](model-inference.md) |

## 入门

- [安装与快速开始](getting-started.md)：源码环境、后端选择、第一个示例。
- [示例与教程](samples.md)：向量加法、张量、共享内存及半精度用例。

## 编程指南

- [Ruda 编程指南](programming-guide.md)：主机与设备、执行层级、内存、同步及安全边界。
- [张量与框架指南](tensor-framework.md)：设备张量、领域库分发、融合与自动微分。
- [张量实用示例](tensor-recipes.md)：矩阵乘、布局、dtype 选择与梯度的完整 CPU 示例。
- [后端选择与组合](backend-composition.md)：CUDA／ROCm／WGPU 选择、本地路由与远程执行。
- [数据管线与模型存储](data-and-storage.md)：样本组批、权重保存与检查点格式导入。
- [训练与状态保存](training.md)：训练步、FP32 主参数／累积、混合存储 checkpoint、token 加权副本、可微分集合通信和学习率调度。
- [Rank、设备与分布式训练](distributed-training.md)：显式 rendezvous、设备映射、集合调用顺序、加权梯度与逐 rank 恢复。
- [架构组件与 Python Muon](architecture-training.md)：mHC、DSA／CSA／HCA、压缩 KV 缓存与混合模型组合。
- [LoRA 与 NF4 微调](finetuning.md)：准确目标选择、流式权重、因果监督、token 加权累积、适配器与断点恢复。
- [通用 PyTorch 模型编译](model-compiler.md)：AOT 前向／反向、原生分段、配置及缓存所有权。
- [固定地址 PyTorch 子图](static-pytorch-graphs.md)：显式 GraphOp、输出生命周期、工作区复用与一阶训练。
- [模型加载与推理](model-inference.md)：ruLLM、文本与图片输入、采样、AWQ 和连续批处理。

## 编译与底层执行

- [编译器指南](compiler-guide.md)：Rust Kernel 前端、IR、CUDA C++／NVRTC 和直接 PTX。
- [PTX 后端参考](ptx.md)：目标配置、编译产物、支持边界与错误行为。
- [全栈自动调优](stack-autotuning.md)：已接入算子、离线校准、计时、缓存身份与策略参数。

## API 参考

- [全栈 Crate 与 API 索引](api-reference.md)：整个 workspace 的包名、Rust 导入名、职责与源码入口。
- [Runtime API](runtime-api.md)：设备客户端、内存、提交、回读与同步。
- [Driver API 与后端](driver-api.md)：后端类型、设备选择和运行时接入。
- [原生 PyTorch API](native-pytorch-api.md)：设备／组件版本、shape／dtype 契约、归一化、优化器、流、注意力、序列训练与量化。
- [计算库参考](libraries/README.md)：按领域选择库、Cargo feature 与接口入口。

## 计算库

| 库 | 手册 |
| --- | --- |
| ruBLAS | [线性代数与分组矩阵乘](libraries/rublas.md) |
| ruDNN | [神经网络算子与 MoE](libraries/rudnn.md) |
| ruPRIM | [归约、扫描与索引](libraries/ruprim.md) |
| ruFFT | [快速傅里叶变换](libraries/rufft.md) |
| ruRAND | [随机数生成](libraries/rurand.md) |
| ruSPARSE | [稀疏计算](libraries/rusparse.md) |
| ruCCL | [集合通信](libraries/ruccl.md) |

## 调试与兼容性

- [调试与诊断](debugging.md)：编译错误、异步错误、缓存与数值检查。
- [兼容性指南](compatibility.md)：CUDA 开发概念对照、后端差异、API 与编译路径边界。
- [贡献指南](CONTRIBUTING.md)：问题反馈与开发约定。

首次使用可按“快速开始 → 编程指南 → 所需计算库”阅读；开发后端或内核时，再查阅编译器与 API 参考。

- [Muon 与显式 Muon + AdamW 参数分组（实验性）](muon.md)
