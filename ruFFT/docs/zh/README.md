# ruFFT

[English](../../README.md) | **简体中文**

Ruda 快速傅里叶变换库。

- Cargo package：`ruFFT`
- Rust crate：`rufft`

## 功能

| Feature | 算子 |
| --- | --- |
| `tensor` | 设备张量上的实数 FFT 与逆变换 |

通过 `rufft::tensor::{rfft, irfft}` 分配变换输出，或使用 `rfft_launch`、`irfft_launch` 直接操作设备绑定。

## 快速开始

在 RUDA 工作区中构建：

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruFFT --no-default-features --features std,tensor
```

## 文档

- [使用手册](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/rufft.md)
- [环境配置](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/getting-started.md)
- [Cargo features](../../Cargo.toml) · [模块入口](../../src/lib.rs)

## ruFFT 用户指南

[计算库](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/README.md) · [张量框架](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/tensor-framework.md) · [English](../../README.md)

ruFFT 在设备上计算实数信号的 FFT 和逆变换。使用 `rufft::tensor` 接口可由库分配输出；已有设备绑定时使用底层 `rfft_launch`、`irfft_launch`。

### 1. 配置依赖

Cargo package 名为 `ruFFT`，Rust 导入名为 `rufft`。`tensor` feature 启用设备张量接口。以下应用目录与 `RUDA` 源码目录同级；NVIDIA 环境配置见[快速开始](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/getting-started.md)。

```toml
[dependencies]
rufft = { package = "ruFFT", path = "../RUDA/ruFFT", default-features = false, features = ["std", "tensor"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

### 2. 正变换与逆变换

下面的 `src/main.rs` 上传四点 F32 信号，计算频谱，再还原输入。在应用目录执行 `cargo run`：

```rust
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use rufft::tensor::{irfft, rfft};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let input = vec![1.0f32, 0.0, -1.0, 0.0];
    let signal = from_data::<CudaRuntime>(TensorData::new(input.clone(), [4]), &device);
    let (real, imag) = rfft(signal, 0, None);

    let re = into_data_sync(real.clone()).to_vec::<f32>()?;
    let im = into_data_sync(imag.clone()).to_vec::<f32>()?;
    for (actual, expected) in re.iter().zip([0.0f32, 2.0, 0.0]) {
        assert!((actual - expected).abs() < 1e-5);
    }
    assert!(im.iter().all(|value| value.abs() < 1e-5));

    let restored = irfft(real, imag, 0, Some(input.len()));
    let restored = into_data_sync(restored).to_vec::<f32>()?;
    for (actual, expected) in restored.iter().zip(input) {
        assert!((actual - expected).abs() < 1e-5);
    }
    println!("real={re:?}, imag={im:?}, restored={restored:?}");
    Ok(())
}
```

四点信号对应三个频点：DC、一个正频率点和 Nyquist 点。实部与虚部分别存储，不使用交错复数布局。正变换不归一化；逆变换包含实际 FFT 长度的归一化，不需要调用方再除以长度。

示例中的 `clone()` 保留张量句柄供逆变换使用；`into_data_sync` 等待回读并返回主机数据，回读失败时 panic。

### 3. 参数与输出形状

| 接口 | 参数 | 返回值 |
| --- | --- | --- |
| `rfft(signal, dim, n)` | 信号、从 0 开始的轴编号、可选请求长度 | `(real, imag)` 两个设备张量 |
| `irfft(real, imag, dim, n)` | 实部、虚部、轴编号、可选输出长度 | 实数设备张量 |

设备计算使用 F32；请传入 F32 信号和频谱。`dim` 必须小于输入 rank，逆变换的实部和虚部须同 shape、同 dtype、同设备。接口返回张量而不是 `Result`；参数断言或启动失败会 panic。

设请求长度为 n，实际 FFT 长度为不小于 n 的最小 2 的幂 N：

- `rfft` 的 n 为 `None` 时取输入在 dim 上的长度；指定 n 时先截断或补零到 n，再补零到 N。
- 正变换输出在 dim 上的长度为 `N / 2 + 1`，其他维度不变。
- `irfft` 的 n 为 `None` 时取 `2 × (频点数 - 1)`；指定 n 时返回该长度。
- 逆变换先将实部、虚部截断或补零到 `N / 2 + 1` 个频点，计算 N 点逆变换，最后裁剪到 n。
- 请求长度须至少为 2；一个点的变换不能通过当前设备内核的长度检查。

| 输入／调用 | 实际变换长度 | 输出在 dim 上的长度 |
| --- | --- | --- |
| 长度 8，`rfft(..., None)` | 8 | 5 |
| 长度 8，`rfft(..., Some(6))` | 8，输入只保留前 6 点，其余补零 | 5 |
| 长度 6，`rfft(..., None)` | 8，尾部补零 | 5 |
| 5 个频点，`irfft(..., None)` | 8 | 8 |
| 5 个频点，`irfft(..., Some(6))` | 8 | 6 |

因此 `Some(6)` 不表示执行六点 DFT。对长度 6 的原始信号做往返变换时，逆变换显式传入 `Some(6)`，才能去掉正变换补入的两个尾点。

### 4. 批量与多维输入

除 dim 以外的轴作为批量维度。例如 `[batch, channels, 1024]` 输入沿 dim=2 变换，实部和虚部都为 `[batch, channels, 513]`。

下面的函数复用上一节的 `rfft` 导入，接收已有设备张量，不执行主机回读：

```rust
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};

fn batched_rfft<R: Runtime>(
    signals: RudaTensor<R>,
    dim: usize,
    length: usize,
) -> (RudaTensor<R>, RudaTensor<R>) {
    rfft(signals, dim, Some(length))
}
```

需要对多个轴变换时，`rfft` 的一次调用只处理指定轴，并不自动执行整个多维 FFT。不要把已分开的实部、虚部分别再次当作完整复数信号传入 `rfft`。

### 5. 缓冲区与执行

张量接口负责输出分配以及需要的补零、裁剪。实际长度大于 4096 时自动使用分阶段路径；调用方式不变。需要自己管理输出缓冲区时，底层启动接口接收 client、输入／输出 `TensorBinding`、dim 和 `StorageType`，返回 `Result<(), LaunchError>`；由调用方按实际 N 准备匹配的输出布局。

已有缓冲区但不想物化补零数据时，使用以下两个入口：

| 接口 | 额外长度参数 |
| --- | --- |
| `rfft_launch_padded(client, signal, real, imag, dim, signal_len, dtype)` | 只读取前 signal_len 个信号元素，其后视为零；N 从输出频谱长度推导 |
| `irfft_launch_padded(client, real, imag, signal, dim, spec_bins, dtype)` | 只读取前 spec_bins 个频点，其后视为零；N 从输出信号长度取得 |

N 必须是至少为 2 的 2 的幂，实部和虚部 shape 相同。`signal_len` 不能超过输入轴长度或 N；`spec_bins` 至少为 1，且不能超过输入频谱轴长度或 `N / 2 + 1`。输出缓冲区仍须完整分配；这两个入口省去的是输入尾部补零的物化复制。

接口参考：[张量入口](../../src/tensor.rs)、[正变换启动](../../src/fft/rfft.rs)、[逆变换启动](../../src/fft/irfft.rs)。
