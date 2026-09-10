# 张量与框架指南

[文档首页](README.md) · [计算库](libraries/README.md) · [编程指南](programming-guide.md) · [English](../en/tensor-framework.md) | [日本語](../ja/tensor-framework.md) | [Deutsch](../de/tensor-framework.md) | [Русский](../ru/tensor-framework.md)

## 1. 层级

| 层 | 组件 | 职责 |
| --- | --- | --- |
| 公共数据与契约 | ruda-core | dtype、shape、设备及编译相关契约 |
| 设备张量 | ruda-kernel::tensor | 存储、元数据、分配与布局操作 |
| 设备 Backend | ruda-tensor-device | 张量操作到领域库的分发 |
| 张量 API | ruda-tensor | 面向 Backend 的张量接口 |
| 融合 | ruda-fusion | 运算融合层 |
| 自动微分 | ruda-autodiff | 自动微分层 |
| 模型与训练组件 | ruda-model、ruda-nn、ruda-optim、ruda-store、ruda-dataset | 模型、网络模块、优化器、存储与数据 |

## 2. 设备张量

`RudaTensor<R>` 包含 client、handle、meta、device、dtype、qparams。存储句柄与 shape／strides 分开，量化参数也单独保存。

低层算子应验证输入是否位于同一设备、dtype 是否符合计算路径、量化数据是否携带正确参数。连续布局、转置视图和物化复制不能混为一谈。

已有分配、连续化、reshape、permutation、transfer、readback 等模块见[设备张量入口](../../ruda-kernel/src/tensor/mod.rs)。

## 3. NVIDIA Backend

`ruda-tensor-device/cuda` 启用 `ruda_tensor_device::cuda`。

不启用 `cuda-fusion` 时，`Cuda<F, I>` 是 `DeviceBackend<CudaRuntime, F, I, u8>` 的别名；启用后使用融合包装层。默认 F 为 f32、I 为 i32。定义见 [cuda.rs](../../ruda-tensor-device/src/cuda.rs)。

这是张量 Backend，不是 CUDA Driver API 句柄。选择后端还需分别确认所需算子的 dtype 与功能范围。

## 4. 领域库分发

矩阵运算进入 ruBLAS，神经网络算子进入 ruDNN，归约与索引进入 ruPRIM，FFT 和随机数分别进入 ruFFT、ruRAND。设备 Backend 的分发组织见 [dispatch](../../ruda-tensor-device/src/dispatch)。

## 5. 稀疏、量化与批量回读

`ruda_tensor::api::CsrTensor<B>` 通过 `SparseOps` 组合稀疏结构和浮点值张量，提供稀疏／稠密乘法、转置、加法、gather、scatter-add 与 sampled 计算。它不同于领域库的 `rusparse::tensor::CsrTensor<R>`：前者泛型为 Backend，后者为 Runtime。公开方法和数据契约见[稀疏手册](libraries/rusparse.md)。

量化链路已有多维 block scale、非末轴打包与尾包处理、量化布局变换、部分索引以及融合量化回读实现。不要将逻辑 shape 当作打包存储 shape，也不要将 FP8／FP4 编码当作整数值转换。相关入口见 [Kernel 量化](../../ruda-kernel/src/quantization)、[量化张量布局](../../ruda-kernel/src/tensor/contiguous.rs)和[融合事务](../../ruda-fusion/src/ops/transaction.rs)。不同 scheme 的操作覆盖并不相同。

批量回读按实际设备与 stream 组织描述符，见[张量事务](../../ruda-kernel/src/tensor/transaction.rs)。

## 6. 训练与模型推理

- [训练与状态保存](training.md)：配置自动微分 Backend、更新参数、累积梯度及保存恢复训练状态。
- [模型加载与推理](model-inference.md)：加载本地权重、构造对话提示、采样生成及图片输入。
