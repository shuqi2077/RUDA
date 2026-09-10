# ruPRIM 用户指南

[计算库](README.md) · [张量框架](../tensor-framework.md) · [English](../../en/libraries/ruprim.md)

ruPRIM 提供设备张量归约、累积扫描、逐元素计算和索引操作。本页使用 `RudaTensor<R>` 接口，参数中的 R 是设备 Runtime。

## 1. 配置依赖

Cargo package 名为 `ruPRIM`，Rust 导入名为 `ruprim`。按所需操作启用 feature：

| feature | 接口 |
| --- | --- |
| `tensor-reduce` | `ruprim::reduce::tensor`：全张量、指定轴归约 |
| `tensor-reduce-autotune` | 归约策略自动调优，同时启用 tensor-reduce |
| `tensor-scan` | `ruprim::scan`：累积和、积、最小值、最大值 |
| `elementwise` | `ruprim::elementwise`：逐元素计算 |
| `indexing` | `ruprim::indexing`：选择、切片、gather、scatter；同时启用 elementwise |

以下应用目录与 `RUDA` 源码目录同级；NVIDIA 环境配置见[快速开始](../getting-started.md)。

```toml
[dependencies]
ruprim = { package = "ruPRIM", path = "../RUDA/ruPRIM", default-features = false, features = ["std", "tensor-reduce", "tensor-scan", "indexing"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

## 2. 求和、逐行归约与扫描

下面的 `src/main.rs` 使用 F32 矩阵 `[[1, 2, 3], [4, 5, 6]]`，计算总和、逐行均值、逐行最大值索引和逐行前缀和。在应用目录执行 `cargo run`：

```rust
use ruda_core::tensor::{DType, TensorMetadata, data::TensorData};
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use ruprim::reduce::{
    components::instructions::ReduceOperationConfig,
    tensor::{KernelReduceStrategy, SumStrategy, reduce_dim, sum},
};
use ruprim::scan::cumsum;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let input = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]),
        &device,
    );
    let total = sum(
        input.clone(),
        SumStrategy::Chained(KernelReduceStrategy::Unspecified),
    )?;
    assert_eq!(total.shape().as_slice(), &[1]);
    assert_eq!(into_data_sync(total).to_vec::<f32>()?, [21.0]);

    let means = reduce_dim(
        input.clone(), None, 1,
        KernelReduceStrategy::Unspecified, ReduceOperationConfig::Mean,
    )?;
    assert_eq!(means.shape().as_slice(), &[2, 1]);
    assert_eq!(into_data_sync(means).to_vec::<f32>()?, [2.0, 5.0]);

    let indices = reduce_dim(
        input.clone(), Some(DType::I32), 1,
        KernelReduceStrategy::Unspecified, ReduceOperationConfig::ArgMax,
    )?;
    assert_eq!(into_data_sync(indices).to_vec::<i32>()?, [2, 2]);

    let prefix = cumsum(input, 1);
    assert_eq!(into_data_sync(prefix).to_vec::<f32>()?, [1., 3., 6., 4., 9., 15.]);
    Ok(())
}
```

`sum` 归约全部元素，输出 shape 为 `[1]`。`reduce_dim` 只处理指定轴；通常保留 rank，将该轴长度改为 1。因此逐行均值为 `[2, 1]`，而不是 `[2]`。

## 3. 归约操作和策略

张量接口的完整参数顺序：

| 接口 | 用途 |
| --- | --- |
| `sum(tensor, strategy)` | 全张量求和 |
| `sum_fallback(tensor, strategy)` | 全张量求和；OneShot 缺少所需原子加能力时改用 Chained |
| `reduce(tensor, output_dtype, strategy, config)` | 依次归约所有轴，返回 shape 为 [1] 的张量 |
| `reduce_dim(tensor, output_dtype, dim, strategy, config)` | 归约指定轴 |

`config` 使用 `ReduceOperationConfig`：

| 操作 | `reduce_dim` 输出 |
| --- | --- |
| `Sum`、`Prod`、`Mean` | 和、积、均值；所选轴长度为 1 |
| `Min`、`Max`、`MaxAbs` | 最小值、最大值、最大绝对值；所选轴长度为 1 |
| `ArgMin`、`ArgMax` | 所选轴内从 0 开始的索引；轴长度为 1 |
| `TopK(k)` | 所选轴的前 k 大值；轴长度为 k |
| `ArgTopK(k)` | 对应的轴内索引；轴长度为 k |

索引归约必须显式传入整数输出类型，例如 `Some(DType::I32)`。值归约传入 `None`，返回输入 dtype；不要把 `output_dtype` 当作通用类型转换参数。F16／BF16 的 Sum、Prod、Mean 在该路径中使用 FP32 累计后写回输入 dtype。轴编号须有效；TopK 的 k 应在 `1..=轴长度` 范围内。TopK 用 `reduce_dim`，不要对它使用会将最终 shape 重设为 `[1]` 的全轴 `reduce`。

`SumStrategy` 的选择：

- `OneShot(ruda_count)`：显式给出工作组数，必须大于零；需要输入 dtype 的原子加能力。
- `Chained(KernelReduceStrategy::Unspecified)`：使用分步归约，不要求全局求和的原子加；示例使用此策略。
- `Autotune`：启用 `tensor-reduce-autotune` 后可用。未启用时默认是 `OneShot(4)`，启用后默认是 `Autotune`。

`KernelReduceStrategy` 可选 `Unspecified`、`Specific(ReduceStrategy)`，以及启用调优后的 `Autotune`。要固定底层策略时使用 Specific；不启用调优时默认 Unspecified。

归约接口返回 `Result<RudaTensor<R>, ReduceError>`。例如轴越界返回 `InvalidAxis`，OneShot 缺少原子加返回 `MissingAtomicAdd`。`sum_fallback` 只替换上述 OneShot 原子加不支持的情况，不会切换到 CPU。

## 4. 累积扫描

四个入口的参数都是 `(tensor, dim)`：

| 函数 | 一行输入 [3, 1, 2] 的结果 |
| --- | --- |
| `cumsum` | [3, 4, 6] |
| `cumprod` | [3, 3, 6] |
| `cummin` | [3, 1, 1] |
| `cummax` | [3, 3, 3] |

这些是包含当前位置的前缀操作，返回与输入相同的 shape 和 dtype；不同批量行分别计算。dim 从 0 开始，必须在输入 rank 范围内。扫描直接返回张量，不返回 `Result`。

当前算法为每个输出位置读取对应前缀，一条长度 n 的序列总读取量为 O(n²)。长轴扫描应计入这部分成本，而不是只按输入元素数估算工作量。

## 5. 选择、gather、切片与整数幂

将下面的函数加入同一文件，并在 main 返回前调用 `indexing_example(&device)?;`。它复用前面的导入：

```rust
fn indexing_example(device: &CudaDevice) -> Result<(), Box<dyn std::error::Error>> {
    use ruda_core::tensor::element::Scalar;
    use ruprim::{
        elementwise::binary::integer_power,
        indexing::{gather, select, slice},
    };

    let input = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]),
        device,
    );
    let columns = from_data::<CudaRuntime>(
        TensorData::new(vec![2i32, 0], [2]), device,
    );
    let selected = select(input.clone(), 1, columns);
    assert_eq!(into_data_sync(selected).to_vec::<f32>()?, [3., 1., 6., 4.]);

    let positions = from_data::<CudaRuntime>(
        TensorData::new(vec![2i32, 0, 1, 1], [2, 2]), device,
    );
    let gathered = gather(1, input.clone(), positions);
    assert_eq!(into_data_sync(gathered).to_vec::<f32>()?, [3., 1., 5., 5.]);

    let sliced = slice(input.clone(), &[0..2, 1..3]);
    assert_eq!(into_data_sync(sliced).to_vec::<f32>()?, [2., 3., 5., 6.]);

    let squared = integer_power::scalar(input, Scalar::Int(2));
    assert_eq!(into_data_sync(squared).to_vec::<f32>()?, [1., 4., 9., 16., 25., 36.]);
    Ok(())
}
```

| 接口 | 索引与输出 |
| --- | --- |
| `select(tensor, dim, indices)` | indices 是一维整数张量；同一组选取位置应用于所有其他维度，输出所选轴长度为 indices 长度 |
| `gather(dim, tensor, indices)` | dim 是第一个参数；按每个输出位置选择输入值，输出 shape 等于 indices shape |
| `slice(tensor, ranges)` | 每轴给出半开区间 `start..end`，示例截取所有行的第 1、2 列 |
| `slice_assign(tensor, slices, value)` | slices 使用 `ruda_core::tensor::Slice`；将 value 写入所选区域并返回更新后的张量 |

索引和数据须同设备，索引为从 0 开始的有效整数。对 gather，使用与输入相同 rank、其他轴尺寸相同的索引张量；示例中每一行可以选择不同列。切片范围须位于对应轴内，赋值张量 shape 要匹配所选区域。

整数幂入口为 `ruprim::elementwise::binary::integer_power`：

- `scalar(input, Scalar::Int(exponent))`：全部元素使用同一个整数指数。
- `tensor(base, exponents)`：浮点底数配整数指数张量，形状需可广播，输出为底数 dtype。

负指数可使用 `Scalar::Int(-2)`；不要先把整数指数转成浮点指数。索引及逐元素接口直接返回设备张量；需要保留输入供其他操作使用时像示例一样传入克隆句柄，并接收每次操作返回的张量。

接口参考：[归约](../../../ruPRIM/src/reduce/tensor/base.rs)、[扫描](../../../ruPRIM/src/scan/tensor.rs)、[索引](../../../ruPRIM/src/indexing/mod.rs)、[整数幂](../../../ruPRIM/src/elementwise/binary/integer_power.rs)。
