# 全栈 Crate 与 API 索引

[文档首页](README.md) · [English](../en/api-reference.md)

## 按任务选择入口层

| 任务 | 首选 API | 指南 |
| --- | --- | --- |
| 编写 GPU Kernel | `ruda`、Kernel DSL、运行时客户端 | [编程指南](programming-guide.md) |
| 使用 typed tensor 与梯度 | `ruda_tensor::api::Tensor`、`Autodiff<B>` | [张量实用示例](tensor-recipes.md) |
| 选择张量执行目标 | `Host`、`Cuda`、`Rocm`、`Wgpu`、Router、Remote | [后端组合](backend-composition.md) |
| 调用领域算子 | ruBLAS、ruDNN、ruPRIM 等领域库 | [计算库](libraries/README.md) |
| 构建和训练模型 | `Module`、`ruda-nn`、`ruda-optim` | [训练指南](training.md) |
| 半精度存储与 FP32 更新 | `Module::to_dtype`、`Fp32MasterOptimizer`、`GradientsAccumulator::accumulate_with_dtype` | [混合精度训练](training.md#fp32-主参数与混合参数存储) |
| 保留混合存储与待累积梯度 | `TrainingRecord::capture_with_dtypes`、`restore_with_dtypes` | [训练状态](training.md#保存和恢复训练状态) |
| 同步副本或对集合通信求导 | `DataParallel`、`ruda_autodiff::collective` | [分布式训练](training.md#副本训练与可微分张量集合通信) |
| 加载样本或权重 | `Dataset`、`DataLoaderBuilder`、`ModuleSnapshot` | [数据与存储](data-and-storage.md) |
| 运行本地模型推理 | `rullm` | [模型推理](model-inference.md) |

## 包名与源码入口

Cargo 包名、仓库目录名和 Rust 导入名可能不同。例如 ruFFT 位于 `ruFFT/`，安装包名为 `ruda-fft`，导入名为 `rufft`；ruSOLVER 的安装包名为 `ruda-solver`，导入名为 `rusolver`。

下表覆盖解析后的 workspace，包括通过路径依赖加入的 CANN 驱动和测试支持包。包名链接指向 Cargo manifest，源码链接指向 crate 入口。Feature 与版本以 manifest 为准，不能根据目录名推断。测试项目与原生 PyTorch 扩展不是普通的 crates.io 安装目标。

## Kernel、编译器与运行时

| Cargo 包名 | Rust 导入名／源码 | 职责 |
| --- | --- | --- |
| [`ruda`](../../ruda/Cargo.toml) | [`ruda`](../../ruda/src/lib.rs) | GPU Kernel 门面与运行时重导出；编写 GPU 内核的入口 |
| [`ruda-core`](../../ruda-core/Cargo.toml) | [`ruda_core`](../../ruda-core/src/lib.rs) | 共享 IR、张量数据、dtype、设备与内存契约 |
| [`ruda-compiler`](../../ruda-compiler/Cargo.toml) | [`ruda_compiler`](../../ruda-compiler/src/lib.rs) | 内核前端与目标代码生成 |
| [`ruda-kernel`](../../ruda-kernel/Cargo.toml) | [`ruda_kernel`](../../ruda-kernel/src/lib.rs) | Rust Kernel DSL、启动接口和设备张量工具 |
| [`ruda-runtime`](../../ruda-runtime/Cargo.toml) | [`ruda_runtime`](../../ruda-runtime/src/lib.rs) | 计算客户端／服务端、存储、流与调度 |
| [`ruda-kernel-macros`](../../ruda-kernel-macros/Cargo.toml) | [`ruda_kernel_macros`](../../ruda-kernel-macros/src/lib.rs) | 将 Kernel Rust 转为 IR 的过程宏 |
| [`ruda-ir-macros`](../../ruda-ir-macros/Cargo.toml) | [`ruda_ir_macros`](../../ruda-ir-macros/src/lib.rs) | IR 实现的派生宏 |

## 驱动与昇腾设备程序

| Cargo 包名 | Rust 导入名／源码 | 职责 |
| --- | --- | --- |
| [`ruda-driver-cuda`](../../ruda-driver-cuda/Cargo.toml) | [`ruda_driver_cuda`](../../ruda-driver-cuda/src/lib.rs) | CUDA 设备、分配、编译与启动 |
| [`ruda-driver-hip`](../../ruda-driver-hip/Cargo.toml) | [`ruda_driver_hip`](../../ruda-driver-hip/src/lib.rs) | AMD HIP 运行时适配 |
| [`ruda-hip-sys`](../../ruda-hip-sys/Cargo.toml) | [`ruda_hip_sys`](../../ruda-hip-sys/src/lib.rs) | 底层 HIP 运行时绑定 |
| [`ruda-driver-wgpu`](../../ruda-driver-wgpu/Cargo.toml) | [`ruda_driver_wgpu`](../../ruda-driver-wgpu/src/lib.rs) | WGPU 初始化、存储和计算执行 |
| [`ruda-driver-cpu`](../../ruda-driver-cpu/Cargo.toml) | [`ruda_driver_cpu`](../../ruda-driver-cpu/src/lib.rs) | CPU Kernel 运行时驱动；区别于 Host 张量后端 |
| [`ruda-driver-cann`](../../ruda-driver-cann/Cargo.toml) | [`ruda_driver_cann`](../../ruda-driver-cann/src/lib.rs) | 动态加载的 CANN AscendCL 接口 |
| [`ruda-ascend-kernels`](../../ruda-ascend-kernels/Cargo.toml) | [`ruda_ascend_kernels`](../../ruda-ascend-kernels/src/lib.rs) | Rust 昇腾设备程序与指令降级 |

## 领域计算库

| Cargo 包名 | Rust 导入名／源码 | 职责 |
| --- | --- | --- |
| [`rublas`](../../ruBLAS/Cargo.toml) | [`rublas`](../../ruBLAS/src/lib.rs) | 矩阵／向量算子与后端分发 |
| [`ruDNN`](../../ruDNN/Cargo.toml) | [`rudnn`](../../ruDNN/src/lib.rs) | 注意力、卷积、池化、归一化与 MoE |
| [`ruPRIM`](../../ruPRIM/Cargo.toml) | [`ruprim`](../../ruPRIM/src/lib.rs) | 归约、扫描、索引与逐元素内核 |
| [`ruda-fft`](../../ruFFT/Cargo.toml) | [`rufft`](../../ruFFT/src/lib.rs) | 傅里叶变换 |
| [`ruRAND`](../../ruRAND/Cargo.toml) | [`rurand`](../../ruRAND/src/lib.rs) | 随机采样与分布 |
| [`ruSPARSE`](../../ruSPARSE/Cargo.toml) | [`rusparse`](../../ruSPARSE/src/lib.rs) | 稀疏格式与算子 |
| [`ruTENSOR`](../../ruTENSOR/Cargo.toml) | [`rutensor`](../../ruTENSOR/src/lib.rs) | 张量收缩、置换与归约 |
| [`ruCCL`](../../ruCCL/Cargo.toml) | [`ruccl`](../../ruCCL/src/lib.rs) | 集合通信算法 |
| [`ruda-solver`](../../ruSOLVER/Cargo.toml) | [`rusolver`](../../ruSOLVER/src/lib.rs) | 主机科学求解器与可选设备求解内核 |
| [`ruintegrate`](../../ruINTEGRATE/Cargo.toml) | [`ruintegrate`](../../ruINTEGRATE/src/lib.rs) | 主机数值积分、ODE 积分与事件定位 |
| [`rublas-host`](../../ruBLAS/host/Cargo.toml) | [`rublas_host`](../../ruBLAS/host/src/lib.rs) | CPU 带步长和批量矩阵乘 |
| [`ruDNN-host`](../../ruDNN/host/Cargo.toml) | [`rudnn_host`](../../ruDNN/host/src/lib.rs) | CPU 神经网络算子实现 |
| [`ruPRIM-host`](../../ruPRIM/host/Cargo.toml) | [`ruprim_host`](../../ruPRIM/host/src/lib.rs) | CPU 张量原语与索引 |
| [`ruFFT-host`](../../ruFFT/host/Cargo.toml) | [`rufft_host`](../../ruFFT/host/src/lib.rs) | CPU 实数傅里叶变换 |
| [`ruRAND-host`](../../ruRAND/host/Cargo.toml) | [`rurand_host`](../../ruRAND/host/src/lib.rs) | 主机随机数生成 |

## 张量、微分与执行组合

| Cargo 包名 | Rust 导入名／源码 | 职责 |
| --- | --- | --- |
| [`ruda-tensor`](../../ruda-tensor/Cargo.toml) | [`ruda_tensor`](../../ruda-tensor/src/lib.rs) | 后端契约与 api feature 下的 api::Tensor |
| [`ruda-tensor-config`](../../ruda-tensor-config/Cargo.toml) | [`ruda_tensor_config`](../../ruda-tensor-config/src/lib.rs) | 自动微分／融合共享配置 |
| [`ruda-tensor-device`](../../ruda-tensor-device/Cargo.toml) | [`ruda_tensor_device`](../../ruda-tensor-device/src/lib.rs) | DeviceBackend、CUDA 适配器与领域库分发 |
| [`ruda-tensor-host`](../../ruda-tensor-host/Cargo.toml) | [`ruda_tensor_host`](../../ruda-tensor-host/src/lib.rs) | Host CPU 张量后端与步长布局 |
| [`ruda-tensor-wgpu`](../../ruda-tensor-wgpu/Cargo.toml) | [`ruda_tensor_wgpu`](../../ruda-tensor-wgpu/src/lib.rs) | WGPU 张量后端与设备初始化 |
| [`ruda-tensor-rocm`](../../ruda-tensor-rocm/Cargo.toml) | [`ruda_tensor_rocm`](../../ruda-tensor-rocm/src/lib.rs) | ROCm 张量适配器 |
| [`ruda-tensor-tch`](../../ruda-tensor-tch/Cargo.toml) | [`ruda_tensor_tch`](../../ruda-tensor-tch/src/lib.rs) | LibTorch 张量适配器 |
| [`ruda-autodiff`](../../ruda-autodiff/Cargo.toml) | [`ruda_autodiff`](../../ruda-autodiff/src/lib.rs) | 梯度图、反向传播与重计算 |
| [`ruda-fusion`](../../ruda-fusion/Cargo.toml) | [`ruda_fusion`](../../ruda-fusion/src/lib.rs) | 张量算子融合规划与执行 |
| [`ruda-tensor-router`](../../ruda-tensor-router/Cargo.toml) | [`ruda_tensor_router`](../../ruda-tensor-router/src/lib.rs) | 本地多后端路由与字节桥接 |
| [`ruda-tensor-remote`](../../ruda-tensor-remote/Cargo.toml) | [`ruda_tensor_remote`](../../ruda-tensor-remote/src/lib.rs) | 远程张量客户端／服务端 |
| [`ruda-communication`](../../ruda-communication/Cargo.toml) | [`ruda_communication`](../../ruda-communication/src/lib.rs) | 传输协议、WebSocket 与张量数据服务 |

## 模型、训练、存储与集成

| Cargo 包名 | Rust 导入名／源码 | 职责 |
| --- | --- | --- |
| [`ruda-model`](../../ruda-model/Cargo.toml) | [`ruda_model`](../../ruda-model/src/lib.rs) | 模块参数、配置、Record 与数据加载器 |
| [`ruda-model-macros`](../../ruda-model-macros/Cargo.toml) | [`ruda_model_macros`](../../ruda-model-macros/src/lib.rs) | Config、Module、Record 派生宏 |
| [`ruda-model-codegen`](../../ruda-model-codegen/Cargo.toml) | [`ruda_model_codegen`](../../ruda-model-codegen/src/lib.rs) | 模型派生宏使用的代码生成器 |
| [`ruda-nn`](../../ruda-nn/Cargo.toml) | [`ruda_nn`](../../ruda-nn/src/lib.rs) | 神经网络层、激活与损失 |
| [`ruda-optim`](../../ruda-optim/Cargo.toml) | [`ruda_optim`](../../ruda-optim/src/lib.rs) | 优化器、梯度累积／裁剪与调度 |
| [`ruda-dataset`](../../ruda-dataset/Cargo.toml) | [`ruda_dataset`](../../ruda-dataset/src/lib.rs) | 索引数据集、数据源与变换 |
| [`ruda-io`](../../ruda-io/Cargo.toml) | [`ruda_io`](../../ruda-io/src/lib.rs) | 主机 I/O 与可选网络下载 |
| [`ruda-store`](../../ruda-store/Cargo.toml) | [`ruda_store`](../../ruda-store/src/lib.rs) | 模型快照、Rudapack、safetensors 与 PyTorch 导入 |
| [`ruda-llm`](../../ruLLM/Cargo.toml) | [`rullm`](../../ruLLM/src/lib.rs) | 模型加载与自回归推理 |
| [`ruda-torch-native`](../../ruda-torch/Cargo.toml) | [`ruda_torch_native`](../../ruda-torch/src/lib.rs) | Python ruda_torch 包的原生 cdylib；源码构建组件 |

## 测试与消费者支持

| Cargo 包名 | Rust 导入名／源码 | 职责 |
| --- | --- | --- |
| [`ruda-test-runtime`](../../ruda-test-runtime/Cargo.toml) | [`ruda_test_runtime`](../../ruda-test-runtime/src/lib.rs) | 运行时／内核测试基础设施 |
| [`ruda-test-utils`](../../ruda-test-utils/Cargo.toml) | [`ruda_test_utils`](../../ruda-test-utils/src/lib.rs) | 内核测试辅助 |
| [`ruda-facade-consumer`](../../ruda/tests/consumer/Cargo.toml) | [`ruda_facade_consumer`](../../ruda/tests/consumer/src/lib.rs) | 门面 feature 连接的外部消费者测试项目 |
| [`ruda-store-pytorch-tests`](../../ruda-store/pytorch-tests/Cargo.toml) | [`ruda_store_pytorch_tests`](../../ruda-store/pytorch-tests/src/lib.rs) | PyTorch 格式互操作测试 |
| [`ruda-store-safetensors-tests`](../../ruda-store/safetensors-tests/Cargo.toml) | [`ruda_store_safetensors_tests`](../../ruda-store/safetensors-tests/src/lib.rs) | Safetensors 格式互操作测试 |

## 查找具体方法

原生 Python 方法见 [PyTorch API 参考](native-pytorch-api.md)、[模型编译](model-compiler.md)、[静态图](static-pytorch-graphs.md)和 [LoRA／NF4 微调](finetuning.md)。算子／应用选择共享[全栈自动调优策略](stack-autotuning.md)。

[架构指南](architecture-training.md)介绍 mHC、压缩注意力／缓存和 Python Muon；[分布式训练指南](distributed-training.md)介绍显式设备、rendezvous、副本初始化、加权归约和逐 rank 恢复。

Typed tensor 方法位于 [ruda-tensor/src/api](../../ruda-tensor/src/api)，后端 trait 位于 [ruda-tensor/src/backend](../../ruda-tensor/src/backend)，领域接口可从各计算库手册进入。使用符号前，先启用暴露该模块的 feature。

设备内存、提交和同步见 [Runtime API](runtime-api.md)，后端初始化和启动契约见 [Driver API](driver-api.md)。模型参数与 Record 类型从 [ruda-model 导出入口](../../ruda-model/src/lib.rs) 查找，不要与编译器中名称相近的 IR 类型混用。
