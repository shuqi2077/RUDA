# Ruda 文档

[文档目录](../README.md) · [English](../en/README.md)

从第一个 GPU Kernel，到领域计算库、张量训练和本地模型推理。按任务开始，再查阅概念指南与 API 参考。

## 从这里开始

| 我想做什么 | 阅读路径 |
| --- | --- |
| 运行第一个 GPU Kernel | [安装与快速开始](getting-started.md) → [编程指南](programming-guide.md) |
| 使用矩阵、稀疏或神经网络算子 | [计算库](libraries/README.md) → [张量与框架](tensor-framework.md) |
| 训练模型、累积梯度与保存状态 | [训练与状态保存](training.md) |
| 加载本地模型、生成文本或处理图片 | [模型加载与推理](model-inference.md) |

## 入门

- [安装与快速开始](getting-started.md)：源码环境、后端选择、第一个示例。
- [示例与教程](samples.md)：向量加法、张量、共享内存及半精度用例。

## 编程指南

- [Ruda 编程指南](programming-guide.md)：主机与设备、执行层级、内存、同步及安全边界。
- [张量与框架指南](tensor-framework.md)：设备张量、领域库分发、融合与自动微分。
- [训练与状态保存](training.md)：训练步、梯度累积、学习率调度、保存与恢复。
- [模型加载与推理](model-inference.md)：ruLLM、文本与图片输入、采样、AWQ 和连续批处理。

## 编译与底层执行

- [编译器指南](compiler-guide.md)：Rust Kernel 前端、IR、CUDA C++／NVRTC 和直接 PTX。
- [PTX 后端参考](ptx.md)：目标配置、编译产物、支持边界与错误行为。

## API 参考

- [Runtime API](runtime-api.md)：设备客户端、内存、提交、回读与同步。
- [Driver API 与后端](driver-api.md)：后端类型、设备选择和运行时接入。
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
