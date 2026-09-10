# ruTENSOR

[English](../../README.md) | **简体中文** | [日本語](../ja/README.md) | [Deutsch](../de/README.md) | [Русский](../ru/README.md)

Ruda 设备张量线性代数：张量收缩与 einsum、归约、物理置换及逐元素运算。

- Cargo package：`ruTENSOR`
- Rust crate：`rutensor`
- [中文使用文档](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/rutensor.md)
- [English user guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rutensor.md)

## ruTENSOR 用户指南

[计算库](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/README.md) · [张量框架](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/tensor-framework.md) · [English](../../README.md)

ruTENSOR 提供基于模式标签的张量收缩、归约、物理置换和逐元素运算。输入使用 `RudaTensor<R>`，由应用选择设备 Runtime。它与上层张量框架 `ruda-tensor` 是不同的库。

### 1. 配置依赖

Cargo package 为 `ruTENSOR`，Rust 导入名为 `rutensor`。默认启用 `std` 和设备张量计算，不绑定某一种驱动。以下配置对应应用目录与 `RUDA` 源码目录并列的布局：

```toml
[dependencies]
rutensor = { package = "ruTENSOR", path = "../RUDA/ruTENSOR" }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

### 2. 张量收缩、归约和置换

将以下内容保存为应用的 `src/main.rs`：

```rust
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use rutensor::{einsum, permute, reduce, ReductionOp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let a = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]), &device,
    );
    let b = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 0., 0., 1., 1., 1.], [3, 2]), &device,
    );
    let product = einsum("ik,kj->ij", &[&a, &b])?;
    let row_sums = reduce(&a, &[0, 1], &[0], ReductionOp::Sum)?;
    let transposed = permute(&a, &[1, 0])?;

    println!("product: {:?}", into_data_sync(product).to_vec::<f32>()?);
    println!("row sums: {:?}", into_data_sync(row_sums).to_vec::<f32>()?);
    println!("transpose: {:?}", into_data_sync(transposed).to_vec::<f32>()?);
    Ok(())
}
```

`einsum("ik,kj->ij", ...)` 对 k 求和，结果形状为 `[2, 2]`。`reduce` 保留标签 0、归约标签 1，结果形状为 `[2]`。`permute` 返回新分配的 `[3, 2]` 张量，而不是共享输入存储的视图。

### 3. einsum 表达式

`einsum(expression, inputs)` 接收一个或多个输入。每个字母表示一个轴，大小写区分；箭头右边决定输出轴及其顺序。

| 表达式 | 运算 |
| --- | --- |
| `ik,kj->ij` | 矩阵乘 |
| `...ik,...kj->...ij` | 带广播批次维的矩阵乘 |
| `abc,cde->abde` | 多维张量收缩 |
| `ij,jk,kl->il` | 三输入收缩 |
| `i,j->ij` | 外积 |
| `ii->i` | 对角线 |
| `ii->` | 迹，输出为零维标量 |
| `ijk->ki` | 沿 j 归约并重排剩余轴 |
| `...i->i` | 归约省略号表示的所有轴 |

- 不同输入中的同名轴必须等长，或其中一方长度为 1。省略号覆盖的轴从右向左对齐广播。
- 同一输入中重复的标签表示对角线，对应轴必须严格等长。
- 输出标签不能重复，且必须出现在输入中。
- 省略 `->` 时，输出先保留省略号轴，再按字母排序保留只出现一次的标签。
- 标量输入对应空标签段；例如 `,ij->ij` 将第一个标量输入乘到矩阵上。
- 输入可以使用非连续布局。计算不会将张量值读回主机。

固定表达式、形状、步幅和 dtype 时，使用 `EinsumPlan::new(expression, descriptors)`，随后反复调用 `execute(inputs)`。需要指定输出与计算精度时，使用 `EinsumPlan::with_options` 或 `einsum_with_options`。

### 4. 描述符与执行计划

`Mode` 是 `i32` 标签。同一个标签在不同张量中表示同一逻辑轴，不要求处于同一物理位置。`TensorDescriptor` 保存形状、以元素为单位的步幅和存储 dtype；`OperandDescriptor` 再关联标签及输入一元变换。

以下函数创建并执行 `D = alpha * A @ B + beta * C` 的计划：

```rust
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use rutensor::{
    ComputeType, DType, OperandDescriptor, OperationDescriptor,
    Plan, Result, TensorDescriptor,
};

fn make_plan<R: Runtime>(
    a: &RudaTensor<R>, b: &RudaTensor<R>, c: &RudaTensor<R>,
    m: usize, n: usize,
) -> Result<Plan> {
    let operation = OperationDescriptor::contraction(
        OperandDescriptor::from_tensor(a, &[0, 2])?,
        OperandDescriptor::from_tensor(b, &[2, 1])?,
        Some(OperandDescriptor::from_tensor(c, &[0, 1])?),
        TensorDescriptor::contiguous(&[m, n], DType::F32)?,
        &[0, 1],
        ComputeType::F32,
    )?;
    Plan::new(operation)
}

fn execute<R: Runtime>(
    plan: &Plan, a: &RudaTensor<R>, b: &RudaTensor<R>, c: &RudaTensor<R>,
    alpha: f64, beta: f64,
) -> Result<RudaTensor<R>> {
    plan.execute(&[a, b, c], &[alpha, beta])
}
```

计划不保存输入缓冲区。后续执行可换成同形状、同步幅、同 dtype 的张量。所有输入必须位于同一设备。

| 构造函数 | 执行输入顺序 | 标量顺序 |
| --- | --- | --- |
| `contraction` | A、B；可选 C | alpha；有 C 时再传 beta |
| `sum_product` | 所有乘积输入；可选 C | alpha；有 C 时再传 beta |
| `reduction` | A；可选 C | alpha；有 C 时再传 beta |
| `permutation` | A | alpha |
| `elementwise_binary` | A、B | alpha、beta |
| `elementwise_trinary` | A、B、C | alpha、beta、gamma |

`Plan::execute` 分配输出。`Plan::execute_into` 接收并返回调用者的输出张量，其形状、步幅和 dtype 必须匹配输出描述符，且缓冲区必须独占，不能与输入或其他视图共享。

使用 `TensorDescriptor::new(extents, strides, dtype)` 指定带间隔或轴重排的输出布局；输出轴不能重叠。显式运算描述符可以通过输出形状指定额外广播轴。

### 5. 归约与逐元素运算

`reduce(input, input_modes, output_modes, operation)` 归约未出现在输出中的标签，不保留被归约轴：

| ReductionOp | 运算 | 空归约结果 |
| --- | --- | --- |
| `Sum` | 求和 | 0 |
| `Product` | 连乘 | 1 |
| `Min` | 最小值 | 正无穷 |
| `Max` | 最大值 | 负无穷 |

`elementwise_binary` 与 `elementwise_trinary` 按标签对齐和广播输入，支持 `BinaryOp::{Add, Mul, Min, Max}`。三元运算按 `(alpha * op(A) op_ab beta * op(B)) op_abc gamma * op(C)` 求值。

通过 `OperandDescriptor::with_unary` 选择 `Identity`、`Negate`、`Abs`、`Sqrt`、`Exp`、`Log`、`Sin`、`Cos`、`Tanh`、`Relu`、`Reciprocal` 或 `Conjugate`。`Log` 是自然对数；实数上的 `Conjugate` 等同于 `Identity`。Min／Max 传播 NaN。

### 6. 精度、存储和错误处理

- 存储类型为 F16、BF16、F32、F64；不接受量化、整数或复数存储。
- `ComputeType::F32` 或 `F64` 控制输入转换、一元变换、乘法、归约及标量运算精度。F64 输入要求 F64 计算。
- 默认情况下，只要存在 F64 输入就使用 F64 计算，否则使用 F32。输入存储类型相同时保留该输出类型；混合输入有 F64 时输出 F64，否则输出 F32。
- 需要更高精度输出时显式指定输出 dtype；提高存储精度不会恢复原始输入已经丢失的精度。
- 输出在最后写回时转换。输入 dtype 与计算 dtype 不同时，库在设备上分配转换缓冲区；相同时直接读取原布局。
- 归约使用工作组内并行树形合并。浮点加法与乘法的顺序可能不同于串行计算。
- 空输出不提交计算；零维输出仍包含一个标量。形状、步幅和地址计算使用平台 `usize`，设备分派同时受后端资源限制。
- 返回的 `Result` 报告表达式、描述符、形状、设备及分派范围错误。设备提交遵循 Runtime 的错误处理，异步执行错误在同步或读回时报告。
- 通用收缩直接遍历归约坐标，不进行多输入收缩路径搜索，也不自动转成 Tensor Core 矩阵乘。存储转换需要额外设备内存。

ruTENSOR 提供 Ruda Rust 接口，不提供 NVIDIA cuTENSOR 的 C ABI。设备必须支持所选存储和计算类型。
