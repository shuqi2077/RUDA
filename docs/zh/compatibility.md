# 兼容性指南

[文档首页](README.md) · [编程指南](programming-guide.md) · [English](../en/compatibility.md) | [日本語](../ja/compatibility.md) | [Deutsch](../de/compatibility.md) | [Русский](../ru/compatibility.md)

## 1. CUDA 开发概念对照

| 熟悉的 CUDA 概念 | Ruda 阅读入口 |
| --- | --- |
| host／device 分工 | 主机 Rust 代码与 Kernel DSL |
| grid／block | RudaCount／RudaDim |
| 一维全局线程位置 | ABSOLUTE_POS |
| 分配与传输 | ComputeClient 的内存和回读接口 |
| Kernel 编译与启动 | ruda-kernel、ruda-compiler、设备运行时 |
| BLAS／DNN／FFT／稀疏等领域库 | ruBLAS／ruDNN／ruFFT／ruSPARSE |
| 集合通信 | ruCCL |

这是概念导航，不是函数一一替换表。各 API 的所有权、参数、同步和错误语义以 Ruda 手册及源码为准。

## 2. CUDA C++ 与 PTX

Ruda 保留 CUDA C++／NVRTC 编译路径，也提供显式选择的直接 PTX 路径。两者均经 NVIDIA 驱动执行。

这里的“保留 CUDA C++ 编译路径”不代表任意 CUDA C++ 工程无修改编译、CUDA Runtime／Driver ABI 完整替代或已有库二进制直接重链接。当前文档也不提供自动转换全部 CUDA 应用的命令。

直接 PTX 支持的是生成器已经实现的 Kernel IR，不是任意 PTX 输入程序。遇到不支持操作不会自动回退另一编译器。

## 3. 后端与 dtype

不同后端对标量类型、原子操作、矩阵指令、内存布局及同步能力的支持不同。先查询设备能力，再核对所需算子的类型与布局要求。

类型在公共 DType 中存在，不等于所有后端的每个算子均可执行该类型。相同 Rust 泛型接口也不保证相同数值舍入或性能。

HIP 是另一套执行接入，不与 PTX 的指令集版本号混用。

## 4. Cargo 与命名

库的展示名、Cargo package 名和 Rust 导入名可能不同，见[计算库索引](libraries/README.md)。功能组合由 features 控制，不能仅凭默认构建代表所有路径。

Kernel 前端使用 `#[ruda]`、RudaCount、RudaDim。

## 5. 版本与数值验证

源码版本、锁文件、features、编译器后端、PTX／SM、驱动和 GPU 共同构成验证配置。

替换计算库调用时，逐项对照布局、转置、索引基准、输入与累计 dtype、归一化、特殊值和同步规则。例如 ruFFT 当前非 2 的幂长度会补齐变换，不能作为任意长度 FFT 的同语义替换。
