# Ruda 编程指南

[文档首页](README.md) · [Runtime API](runtime-api.md) · [计算库](libraries/README.md) · [English](../en/programming-guide.md)

## 1. 主机与设备

主机 Rust 代码负责选择设备、准备输入、构造参数及读取结果。设备 Kernel 描述并行计算，经前端展开为 IR，再由后端编译和执行。

`ruda-kernel::dsl` 是当前通用 Kernel 前端。Kernel 使用 Rust 语法与该前端提供的类型、宏和操作；这不意味着任意 Rust 程序及其标准库都能直接编译到 GPU。

上层张量框架通过领域库分发运算。应用无需为了调用矩阵乘而自行实现线程级 Kernel。

## 2. 执行层级

| Ruda 概念 | 用途 |
| --- | --- |
| `RudaCount` | 一次启动中的工作组数量 |
| `RudaDim` | 每个工作组的执行尺寸 |
| `ABSOLUTE_POS` | 一维逐元素 Kernel 的全局位置 |
| `Array<T>` | Kernel 中的一维数组访问 |
| `Tensor<T>` | 带维度与步长信息的 Kernel 张量访问 |
| `Runtime` | 关联编译器、计算服务和设备类型 |

这些名称来自当前 API；CUDA 概念对照见[兼容性指南](compatibility.md)。公共导出见 [DSL prelude](../../ruda-kernel/src/dsl/prelude.rs)。

## 3. 第一个 Kernel

下面摘自 [ptx-runtime 示例](../../ruda-driver-cuda/examples/ptx_runtime.rs)，完整主机代码和执行检查保留在示例中：

```rust
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
fn add(a: &Array<f32>, b: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}
```

`#[ruda(launch)]` 为当前宏名，相关类型由 `ruda_kernel::dsl::prelude::*` 导入。输出长度用于排除尾部线程；调用者仍需保证输入数组至少具有相同长度。

示例按每组 64 个执行单元、工作组数向上取整启动。64 是该示例的配置，不是所有内核的最优尺寸。

## 4. 内存与参数

主机端通过 `ComputeClient` 创建或分配设备缓冲区，使用句柄构造 Kernel 参数。尺寸必须区分字节数和元素数：

- `client.empty(size)` 的 `size` 是字节数。
- 示例的 `ArrayArg::from_raw_parts(handle, count)` 中，`count` 是数组元素数。
- `RudaTensor<R>` 同时携带存储句柄、shape、strides、dtype、设备和量化参数。

克隆句柄或张量不能理解为复制底层设备数据。需要改变布局时，应使用相应的连续化、复制或变换接口；不能只修改元数据来假装存储已经重排。

## 5. 提交、回读与同步

Kernel 提交与结果可用是不同阶段。主机提交结束不能作为设备计算耗时或执行成功的证据。

`read_one` 等待回读并返回 `Result`；`read_async` 提供异步结果。`sync()` 返回 future，调用者需要等待其完成。`flush()` 用于提交积压命令，不应作为读取结果的替代。

多流访问同一份数据时必须满足生产者与消费者的执行依赖。`set_stream` 是 unsafe 接口，不能仅凭主机变量的生命周期推断设备任务已经完成。

## 6. 安全边界

类型、所有权和借用机制服务于主机端资源与接口约束；底层封装仍需要维护设备执行的安全条件：

- 参数的存储范围、dtype、对齐与布局和 Kernel 访问一致。
- 异步任务使用的数据在任务完成前持续有效。
- 不同线程和流的共享写入具有正确同步。
- 裸参数与 unchecked 启动的调用者满足接口安全约定。

受检查的启动不等于对任意 Kernel 的完整安全证明。当前示例在原始参数构造与启动处保留显式 `unsafe` 块及安全依据。详细入口见 [Runtime API](runtime-api.md)。

## 7. 从 Kernel 到领域库

通用算子优先使用 [ruBLAS](libraries/rublas.md)、[ruDNN](libraries/rudnn.md)、[ruPRIM](libraries/ruprim.md) 等库。下沉到 Kernel 层时，明确输入布局、累计精度和执行配置。
