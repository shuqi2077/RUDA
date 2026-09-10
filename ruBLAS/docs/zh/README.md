# ruBLAS

[English](../../README.md) | **简体中文** | [日本語](../ja/README.md) | [Deutsch](../de/README.md) | [Русский](../ru/README.md)

Ruda 线性代数与矩阵乘法库。

- Cargo package：`rublas`
- Rust crate：`rublas`

## 功能

| Feature | 算子 |
| --- | --- |
| `tensor-matmul` | 矩阵乘法 |
| `tensor-matmul-autotune` | 矩阵乘法自动调优 |
| `tensor-vector` | 向量运算 |
| `tensor-grouped` | 分组矩阵乘法 |
| `tensor-int4` | INT4 权重运算 |

## 快速开始

在 RUDA 工作区中构建：

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p rublas --features tensor-matmul
```

## 文档

- [使用手册](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/rublas.md)
- [环境配置](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/getting-started.md)
- [Cargo features](../../Cargo.toml) · [模块入口](../../src/lib.rs)

## ruBLAS 用户指南

[计算库](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/README.md) · [Runtime API](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/runtime-api.md) · [English](../../README.md)

### 1. 概述与功能入口

ruBLAS 组织线性代数计算及其设备 Tensor 入口。

| feature | 模块 |
| --- | --- |
| `tensor-vector` | `rublas::tensor_vector` |
| `tensor-matmul` | `rublas::tensor_matmul` |
| `tensor-matmul-autotune` | 张量矩阵乘调优入口 |
| `tensor-int4` | `rublas::tensor_int4` |
| `tensor-grouped` | `rublas::tensor_grouped` |

Cargo package 为 `rublas`。通用路径选择所需 feature，并关闭不需要的默认功能。定义见 [Cargo.toml](../../Cargo.toml)。

### 2. 矩阵乘、向量与 INT4

#### 张量矩阵乘

`rublas::tensor_matmul::matmul` 接收 lhs、rhs、可选输出 out、MatmulStrategy 和输出 DType，返回设备张量或 MatmulSetupError。已有输出可通过 out 传入；未传入时由入口创建。

策略包括 Ruda、CmmaResidueFirst、Naive，以及启用 `tensor-matmul-autotune` 后的 Autotune。默认策略随该 feature 改变：未启用时为 Ruda，启用后为 Autotune。Naive 路径处理量化输入时可能在原路径失败后解量化再计算，不等于原生 INT4 计算。

`CmmaResidueFirst` 显式选择先处理不足 K32 尾块的 Tensor Core 路径，无需调用者物化补齐后的矩阵。lhs 和 rhs 必须是相同的非量化 BF16 或 F16，且矩阵维度匹配；设备还需支持该路径使用的矩阵指令。下面的函数返回 F32 输出：

```rust
use ruda_core::tensor::DType;
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use rublas::{
    kernel_ir::definition::MatmulSetupError,
    tensor_matmul::{MatmulStrategy, matmul},
};

fn residue_matmul<R: Runtime>(
    lhs: RudaTensor<R>,
    rhs: RudaTensor<R>,
) -> Result<RudaTensor<R>, MatmulSetupError> {
    matmul(lhs, rhs, None, MatmulStrategy::CmmaResidueFirst, DType::F32)
}
```

输入为 `[M, K]` 和 `[K, N]` 时输出为 `[M, N]`。设置或设备能力不满足时返回 `MatmulSetupError`，不会改用 Naive。

定义见[矩阵乘入口](../../src/tensor_matmul/base.rs)。

#### 向量叉积

`rublas::tensor_vector::cross(lhs, rhs, dim)` 要求指定维度长度为 3，返回设备张量。非末维计算涉及维度置换与连续化。定义见[叉积实现](../../src/tensor_vector/cross.rs)。

#### AWQ INT4

`rublas::tensor_int4::AwqGemm::new` 接收 qweight、qzeros、scales、可选 bias 和 group_size，建立打包权重对象；`forward` 接收 F16 输入并返回 F16 输出或 Int4Error。

对于输入维度 K、输出维度 N、分组大小 G：

| 数据 | dtype | shape |
| --- | --- | --- |
| qweight | I32 打包 | [K, N/8] |
| qzeros | I32 打包 | [K/G, N/8] |
| scales | F16 | [K/G, N] |
| bias，可选 | F16 | [N] |

K、N、G 非零，K 能被 G 整除，N 能被 8 整除。输入最后一维为 K，输出替换为 N；所有操作数需同设备。打包位序必须匹配[AWQ 内核](../../src/tensor_int4/kernel.rs)，不能把任意 INT4 文件直接作为 qweight 使用。

对象和检查定义见[INT4 入口](../../src/tensor_int4/mod.rs)。

### 3. 分组矩阵乘

入口为 `rublas::tensor_grouped::grouped_matmul_nt<R: Runtime>`。它接收 input、weights、row_experts 三个 `RudaTensor<R>`，返回 `Result<RudaTensor<R>, GroupedMatmulError>`。

| 参数 | shape | 要求 |
| --- | --- | --- |
| input | [M, K] | 非量化 F32、F16 或 BF16 |
| weights | [E, N, K] | 与 input 相同 dtype、同一设备 |
| row_experts | [M] | 非量化 U32、同一设备 |
| 输出 | [M, N] | 与 input 相同 dtype |

第 m 行选择编号为 row_experts[m] 的权重矩阵，计算输入行与该矩阵各行的内积。权重的最后两维按转置方式参与计算，不要求调用者先物化转置。

### 4. 分组矩阵乘的数值与边界

- K、N、E 必须大于零；M 可以为零。
- 超出专家范围的行编号表示 padding，输出零行。
- 当前内核采用 FP32 累计，最终转换为输入 dtype。
- 输入和权重需要的连续化由入口处理，可能引入数据复制。
- 相关元素数量不能超过 U32 索引范围。
- 参数不满足约定时返回 `GroupedMatmulError`；异步执行错误还需在回读或同步阶段处理。

源码：[分组矩阵乘入口](../../src/tensor_grouped/mod.rs)、[内核](../../src/tensor_grouped/kernel.rs)。

### 5. 与其他库协作

[ruDNN MoE](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/rudnn.md) 使用分组矩阵乘计算专家投影。量化权重路径位于独立的 `tensor_int4` 模块；不能将普通浮点分组矩阵乘当作 INT4 专家计算。

当前分组内核采用标量累计。矩阵乘策略应按实际 dtype、shape 和后端分别测量。
