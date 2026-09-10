# ruRAND 用户指南

[计算库](README.md) · [Runtime API](../runtime-api.md) · [English](../../en/libraries/rurand.md)

ruRAND 生成均匀、正态和伯努利分布的设备张量。用 `rurand::tensor` 分配并生成新张量，或用 crate 根部的同名函数填充已有设备存储。

## 1. 配置依赖

Cargo package 名为 `ruRAND`，Rust 导入名为 `rurand`。`tensor` feature 启用设备张量入口。以下应用目录与 `RUDA` 源码目录同级；NVIDIA 环境配置见[快速开始](../getting-started.md)。

```toml
[dependencies]
rurand = { package = "ruRAND", path = "../RUDA/ruRAND", default-features = false, features = ["std", "tensor"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

## 2. 生成随机张量

下面的 `src/main.rs` 生成形状为 `[2, 3]` 的 F32 张量，并展示重置种子后重放同一次调用。在应用目录执行 `cargo run`：

```rust
use ruda_core::tensor::DType;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::readback::into_data_sync;
use rurand::tensor::{
    random_bernoulli, random_like_uniform, random_normal, random_uniform,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    rurand::seed(42);
    let uniform = random_uniform::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let first = into_data_sync(uniform.clone()).to_vec::<f32>()?;

    rurand::seed(42);
    let repeated = random_uniform::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let repeated = into_data_sync(repeated).to_vec::<f32>()?;
    assert_eq!(first, repeated);
    assert!(first.iter().all(|&x| (0.0..1.0).contains(&x)));

    let normal = random_normal::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let mask = random_bernoulli::<CudaRuntime>([2, 3].into(), &device, 0.25, DType::F32);
    let noise = random_like_uniform(&uniform, -0.1, 0.1, DType::F32);

    let mask = into_data_sync(mask).to_vec::<f32>()?;
    assert!(mask.iter().all(|&x| x == 0.0 || x == 1.0));
    println!("uniform={first:?}");
    println!("normal={:?}", into_data_sync(normal).to_vec::<f32>()?);
    println!("mask={mask:?}");
    println!("noise={:?}", into_data_sync(noise).to_vec::<f32>()?);
    Ok(())
}
```

`random_like_uniform` 只从参考张量取得 shape 和 device；它分配新张量，不修改参考张量，也不复用参考张量的值。输出 dtype 由最后一个参数显式指定。

`random_bernoulli` 返回所选 dtype 表示的 0 或 1，不自动创建布尔类型掩码；本例返回 F32 的 0.0／1.0。

## 3. 参数与分布

前三个张量入口的共同参数是 `shape: Shape`、`device: &R::Device`，以及末尾的 `dtype: DType`。返回的 `RudaTensor<R>` 具有指定 shape、dtype 和 device。

| 入口及参数顺序 | 分布参数 |
| --- | --- |
| `random_uniform(shape, device, lower_bound, upper_bound, dtype)` | f32 下界与上界 |
| `random_normal(shape, device, mean, std, dtype)` | f32 均值与标准差；std 不是方差 |
| `random_bernoulli(shape, device, probability, dtype)` | f32 的取 1 概率 |
| `random_like_uniform(&reference, lower_bound, upper_bound, dtype)` | 参考张量、f32 下界与上界、输出 dtype |

按分布定义传入有限参数：均匀分布的下界小于上界，正态分布的标准差非负，伯努利概率在 `[0, 1]`。这些入口不会以 `Result` 校验上述参数区间，不要依赖传入非法参数来获得错误返回。

均匀分布先生成 F32 的 `u ∈ [0, 1)`，再计算 `lower_bound + (upper_bound - lower_bound) × u` 并转换到输出 dtype。`[0, 1)` 的 F32 用例不包含 1；其他区间缩放或低精度转换可能将接近上界的值舍入到上界。

正态分布在 FP32 中使用 Box–Muller 变换后转换到输出 dtype。选择 F64 存储不会增加生成器内部的随机浮点精度。伯努利用 `u < probability` 生成值，所以概率 0 输出全零，概率 1 输出全一。

## 4. 种子与调用顺序

`rurand::seed(seed: u64)` 重置共享的主机随机状态。每次生成从该状态取出新种子并推进状态：

- 初始化随机序列时设置一次 seed，然后连续生成不同批次。
- 想重放同一序列时，重设相同 seed，并保持生成调用的顺序、shape、dtype 和设备配置一致。
- 不要在每个训练批次前重设相同 seed，除非就是要重复相同随机输入。
- 多线程并发生成共享此状态，调用的交错顺序会影响各调用拿到的种子；它不是每个线程或设备各自独立的生成器对象。

上面的重放比较在同一进程、同一设备及相同调用参数下进行。与其他随机库或不同后端比较时，不能只用 seed 相同来要求随机数组逐项相同。

## 5. 填充已有缓冲区

需要复用输出分配时，使用 crate 根部的启动接口。以下函数接收已经分配好的 F32 设备张量，在原存储中写入随机值并返回该张量：

```rust
use ruda_kernel::{
    dsl::{Runtime, prelude::LaunchError},
    tensor::RudaTensor,
};

fn fill_uniform<R: Runtime>(
    output: RudaTensor<R>,
    lower: f32,
    upper: f32,
) -> Result<RudaTensor<R>, LaunchError> {
    rurand::random_uniform(
        &output.client,
        lower,
        upper,
        output.clone().binding(),
        output.dtype.into(),
    )?;
    Ok(output)
}
```

该函数不调用张量分配器。若其他张量句柄共享这份存储，它们也会看到写入结果。填充正态或伯努利数据时，同层接口分别为 `rurand::random_normal(&client, mean, std, binding, dtype)` 和 `rurand::random_bernoulli(&client, probability, binding, dtype)`。

张量级入口在启动返回错误时 panic；上面的底层入口返回 `Result<(), LaunchError>`。提交成功不代表 GPU 已执行完毕，读取结果时使用张量回读接口或等待客户端同步。`into_data_sync` 在回读失败时 panic；需要传播回读错误时使用异步 `into_data(tensor).await`。

接口参考：[张量入口](../../../ruRAND/src/tensor/mod.rs)、[种子状态](../../../ruRAND/src/state.rs)、[分布实现](../../../ruRAND/src/distributions/mod.rs)。
