# ruSPARSE

[English](../../README.md) | **简体中文**

本仓库是源码镜像。以下命令需在 [RUDA 主仓库](https://github.com/shuqi2077/RUDA)根目录运行。

Ruda 稀疏计算库，复用公共张量、运行时和设备后端。

## Feature 与入口

| Feature | 入口与依赖 |
| --- | --- |
| 默认（空） | 稀疏格式、校验及主机转换；不引入设备实现 |
| `tensor` | `CsrTensor<R>`、SpMV、SpMM、SpVV、SDDMM、SpGEAM、SpGEMM；由调用者选择公共运行时 |
| `cuda` | `tensor` 加 NVIDIA 驱动，供 CUDA 程序与示例使用 |
| `cuda-tests` | `cuda` 的测试兼容入口 |

## CUDA 示例

```sh
cargo run -p ruSPARSE --features cuda --example csrmv
```

示例执行 `alpha * A * x + beta * y`，同步读回并校验 `[7.0, 2.0, 18.5]`。

## 当前计算契约

- 公共设备路径使用 FP32 值及 U32 索引；数值内核要求 32-lane 子组。
- 通过 `CsrTensor::from_csr` 上传已校验的结构；`with_values` 更新数值，`transpose` 重排当前设备值。操作返回新结果，不覆盖输入。
- SpGEAM／SpGEMM 共享主机符号算法，保留显式零结构；数值阶段在设备执行。结构索引保留主机副本，数值不回读到主机计算。

## ruSPARSE 用户指南

[计算库](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/README.md) · [English](../../README.md)

### 1. 概述

ruSPARSE 负责稀疏计算。

### 2. 数据描述

| 类型 | 用途 |
| --- | --- |
| `CsrMatrix`、`CsrMatrixOwned` | CSR 借用视图与自有存储 |
| `CooMatrix`、`CooMatrixOwned` | COO 表示 |
| `CscMatrix`、`CscMatrixOwned` | CSC 表示 |
| `BsrMatrix`、`EllMatrix` | 块稀疏与 ELL 表示 |
| `DenseMatrix`、`DenseMatrixOwned` | 稠密矩阵描述 |
| `SparseVector` | 稀疏向量 |
| `IndexBase` | Zero 或 One 索引基准 |
| `DenseOrder` | RowMajor 或 ColumnMajor |
| `Operation` | None、Transpose、ConjugateTranspose |

数据的索引基准、布局和操作类型是显式契约，不能只根据数组长度猜测。名称中的 Owned 区分自有数据与借用视图，不表示自动分配通用 GPU 存储。

### 3. 运算组织

现有实现按矩阵向量乘、矩阵乘、稀疏矩阵乘／加、稀疏向量内积、SDDMM 和格式转换组织。执行层包含 `SparsePlan`、`SparseError` 及不同操作的执行结果类型。

CSR 矩阵向量乘采用 alpha、op(A)、x、beta、y 的运算契约；迁移调用时需要同时匹配索引基准、转置方式和数据布局，不是只替换库名前缀。

### 4. CUDA CSR 矩阵向量乘

Cargo package 名为 `ruSPARSE`，Rust 导入名为 `rusparse`。`tensor` feature 启用 `rusparse::tensor`，`cuda` 同时启用设备张量和 CUDA 依赖。在源码根目录运行：

```powershell
cargo run --locked -p ruSPARSE --features cuda --example csrmv
```

这个示例计算 `0.5 × A × x + 2 × y`：

```rust
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use rusparse::{
    CsrMatrix, IndexBase, Operation,
    tensor::{CsrTensor, csrmv},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let host = CsrMatrix::new(
        3, 3,
        &[0, 2, 3, 4],
        &[0, 1, 1, 2],
        &[2., 1., -1., 5.],
        IndexBase::Zero,
    )?;
    let matrix = CsrTensor::<CudaRuntime>::from_csr(host, Operation::None, &device)?;
    let x = from_data(TensorData::new(vec![3f32, 4., 5.], [3]), &device);
    let y = from_data(TensorData::new(vec![1f32, 2., 3.], [3]), &device);
    let output = into_data_sync(csrmv(&matrix, 0.5, x, 2., y)?).to_vec::<f32>()?;
    assert_eq!(output, [7., 2., 18.5]);
    println!("alpha * A * x + beta * y = {output:?}");
    Ok(())
}
```

`CsrMatrix::new` 的参数依次是行数、列数、行偏移、列索引、非零值和索引基准。本例第一行存储两项，第二、第三行各一项；结果为 `[7.0, 2.0, 18.5]`。

`CsrTensor::from_csr` 上传结构和值；转置在此处用 `Operation::Transpose` 指定。`csrmv(&matrix, alpha, x, beta, y)` 的 x 长度为矩阵列数，y 长度为行数；稠密输入非量化、dtype 为 F32，且与矩阵位于同一设备。返回新设备张量，复用矩阵时不必再次上传 CSR 结构。

稀疏／稠密矩阵乘使用 `csrmm(&matrix, operation_b, alpha, b, beta, c, output_order)`，其中 c 为可选张量，输出布局使用 `DenseOrder` 指定。仅更新非零值时，调用 `matrix.with_values(values)`，新 values 为长度 nnz 的 F32 设备张量，结构保持不变。

### 5. 张量框架中的 CSR

`ruda_tensor::api::CsrTensor<B>` 面向实现 `SparseOps` 的 Backend，与上一节的 `rusparse::tensor::CsrTensor<R>` 是不同类型。用 `CsrTensor::<B>::from_data(&data, &device)` 创建，data 类型为该 Backend 的 `B::CsrData`。

| 方法 | 用法 |
| --- | --- |
| `matmul(rhs)`、`transpose_matmul(rhs)` | 稀疏矩阵或其转置乘二维稠密张量 |
| `sparse_matmul(&rhs)` | 稀疏矩阵相乘，生成结果结构 |
| `add(&rhs)`、`add_scaled(&rhs, alpha, beta)` | 稀疏加法或带系数加法 |
| `gather(dense)` | 按当前 CSR 位置提取稠密值，返回一维张量 |
| `scatter_add()` | 将存储值加到对应稠密位置 |
| `mul_dense(rhs)` | 在当前 CSR 位置与稠密矩阵逐项相乘 |
| `sampled_matmul(lhs, rhs)` | 只在当前 CSR 位置计算稠密矩阵乘结果 |
| `sampled_sparse_matmul(&lhs, &rhs)` | 只在当前 CSR 位置计算稀疏矩阵乘结果 |
| `transpose()`、`to_dense()` | 转置或转换为稠密张量 |
| `with_values(values)` | 替换一维值张量并保留结构 |
| `to_data().await` | 异步回读为 `B::CsrData` |

这些计算方法返回 `Result`；框架稀疏运算的值张量必须非量化。

接口参考：[设备 CSR](../../src/tensor/mod.rs)、[框架 CSR](https://github.com/shuqi2077/RUDA/blob/main/ruda-tensor/src/api/sparse.rs)。
