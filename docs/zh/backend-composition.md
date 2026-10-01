# 后端选择与组合

[文档首页](README.md) · [张量实用示例](tensor-recipes.md) · [English](../en/backend-composition.md)

## Kernel 运行时与张量后端

`cargo add ruda --features cuda` 选择 CUDA Kernel 门面，并不把 `ruda` 变成张量、模型和优化器库。使用 `Tensor<B, D>` 时，需要选择张量后端，并启用 `ruda-tensor/api` 或 `api-std`。

| 执行目标 | 张量后端 | 设备 | 选择方式 |
| --- | --- | --- | --- |
| CPU | `ruda_tensor_host::Host` | `HostDevice` | `ruda-tensor-host`，启用 `std`，可选 `simd`／`rayon` |
| NVIDIA CUDA | `ruda_tensor_device::cuda::Cuda<f32, i32>` | `CudaDevice` | `ruda-tensor-device/cuda-default` |
| AMD ROCm | `ruda_tensor_rocm::Rocm<f32, i32>` | `RocmDevice` | `ruda-tensor-rocm` 与 ROCm/HIP 运行时 |
| WGPU | `ruda_tensor_wgpu::Wgpu<f32, i32>` | `WgpuDevice` | 按目标选择 `vulkan`、`metal` 或 `webgpu` |
| LibTorch | `ruda_tensor_tch::LibTorch` | `LibTorchDevice` | 匹配的 LibTorch 安装 |
| 多个本地后端 | `ruda_tensor_router::Router<(B1, B2)>` | `duo::MultiDevice<B1, B2>` | 显式指定设备变体 |
| 远程执行 | `ruda_tensor_remote::RemoteBackend` | `RemoteDevice` | `client` feature 与已启动的服务端 |

仍需相应硬件驱动和目标工具链。启用 Cargo feature 不会安装这些组件，也不会自动提供所有后端。WGPU 路径选择图形计算 API，不是自动在原生 CUDA 和 HIP 之间切换。

本仓库的昇腾入口包括 `ruda-driver-cann`、`ruda-ascend-kernels`、编译器的 Ascend 路径和 `rublas::cann`；它们与上表的 typed tensor 后端入口不同，见[编译器指南](compiler-guide.md)和 [ruBLAS 指南](libraries/rublas.md)。

## CUDA 张量依赖

使用 typed tensor 的项目添加：

```toml
[dependencies]
ruda-tensor = { version = "0.21", default-features = false, features = ["api-std"] }
ruda-tensor-device = { version = "0.21", default-features = false, features = ["cuda-default"] }
```

导入 `ruda_tensor_device::cuda::{Cuda, CudaDevice}`，后端类型选择 `Cuda<f32, i32>`。包含模型和优化器的完整示例见[训练指南](training.md)。

`cuda` 暴露适配器，`cuda-fusion` 用 `Fusion` 包装该适配器，`cuda-default` 则组合 CUDA 融合与默认运行时／后端设施。ROCm 和 WGPU 导出的后端别名是否带融合包装，由其 `fusion` feature 决定。Cargo feature 会合并：其他依赖启用 fusion，也会影响最终别名。

## 在 CPU 与 WGPU 之间迁移张量

本例明确使用 Vulkan，需要兼容的 Vulkan 适配器。创建二进制项目并配置：

```toml
[dependencies]
ruda-tensor = { version = "0.21", default-features = false, features = ["api-std"] }
ruda-tensor-host = { version = "0.21", default-features = false, features = ["std"] }
ruda-tensor-router = "0.21"
ruda-tensor-wgpu = { version = "0.21", features = ["vulkan"] }
```

将以下代码放入 `src/main.rs`，执行 `cargo run`：

```rust
use ruda_tensor::{api::Tensor, DType};
use ruda_tensor_host::{Host, HostDevice};
use ruda_tensor_router::{duo::MultiDevice, Router};
use ruda_tensor_wgpu::{Wgpu, WgpuDevice};

type B = Router<(Host, Wgpu<f32, i32>)>;
type Device = MultiDevice<Host, Wgpu<f32, i32>>;

fn main() {
    let cpu = Device::B1(HostDevice);
    let gpu = Device::B2(WgpuDevice::default());
    let x = Tensor::<B, 1>::from_data([1.0f32, 2.0, 3.0], (&cpu, DType::F32));
    let x_gpu = x.to_device(&gpu);
    let result = (x_gpu.clone() + x_gpu).to_device(&cpu).into_data();
    assert_eq!(result.as_slice::<f32>().unwrap(), &[2.0, 4.0, 6.0]);
}
```

`B1`、`B2` 与后端元组顺序一致。Router 默认设备属于第一个后端；显式变体可明确放置位置。`trio` 和 `quad` 提供三后端、四后端的对应设备枚举。

`Router` 使用 `DirectByteChannel` 和 `ByteBridge`，跨后端迁移经过 `TensorData`，不保证零拷贝或 GPU 点对点传输。`to_device` 显式迁移张量。Router 不会自动选择最快后端、划分模型，或把不支持的算子替换为 CPU 实现。

## 远程张量执行

服务端决定实际执行后端，客户端通过 WebSocket 发送张量操作。学习协议可以使用 CPU 服务端。新建项目并配置：

```toml
[dependencies]
ruda-tensor = { version = "0.21", default-features = false, features = ["api-std"] }
ruda-tensor-host = { version = "0.21", default-features = false, features = ["std"] }
ruda-tensor-remote = { version = "0.21", default-features = false, features = ["client", "server"] }
```

创建 `src/bin/server.rs`：

```rust
use ruda_tensor_host::{Host, HostDevice};

fn main() {
    ruda_tensor_remote::server::start_websocket::<Host>(HostDevice, 3000);
}
```

创建 `src/bin/client.rs`：

```rust
use ruda_tensor::{api::Tensor, DType};
use ruda_tensor_remote::{RemoteBackend, RemoteDevice};

fn main() {
    let device = RemoteDevice::new("ws://127.0.0.1:3000");
    let x = Tensor::<RemoteBackend, 1>::from_data(
        [1.0f32, 2.0, 3.0], (&device, DType::F32),
    );
    let result = (x.clone() + x).into_data();
    assert_eq!(result.as_slice::<f32>().unwrap(), &[2.0, 4.0, 6.0]);
}
```

先执行 `cargo run --bin server`，再在第二个终端执行 `cargo run --bin client`。便捷服务端绑定的是 `0.0.0.0:3000`，而非仅回环地址；内置传输没有配置认证或 TLS。上面的客户端 URL 用于同机连接。

`start_websocket` 自带 Tokio runtime，并阻塞至服务端退出。已有异步 runtime 时使用 `start_websocket_async::<B>(device, port).await`。服务端后端必须实现 `BackendIr`；切换服务端后端也需要对应依赖与硬件。远程张量执行和 ruCCL 分布式集合通信不是同一接口。

## 包装层与职责边界

- 张量图需要梯度时使用 `Autodiff<B>`，见[张量实用示例](tensor-recipes.md)。
- Fusion 优化支持的算子序列，不会补出后端缺失的算子或硬件能力。
- Router 控制本地放置与传输，Remote 将执行放到远程传输之后。
- ruCCL 提供集合通信，而不是远程张量服务端，见 [ruCCL](libraries/ruccl.md)。
- 统一后端接口不表示 dtype、稀疏和量化算子的覆盖完全相同，见[兼容性](compatibility.md)及对应计算库手册。

## API 入口

- [CUDA 别名](../../ruda-tensor-device/src/cuda.rs)、[ROCm 别名](../../ruda-tensor-rocm/src/lib.rs)和 [WGPU 初始化](../../ruda-tensor-wgpu/src/lib.rs)。
- [Router 别名](../../ruda-tensor-router/src/lib.rs)、[设备枚举与字节桥接](../../ruda-tensor-router/src/types.rs)。
- [远程客户端](../../ruda-tensor-remote/src/lib.rs)、[服务端启动](../../ruda-tensor-remote/src/server/base.rs)和 [WebSocket 监听器](../../ruda-communication/src/websocket/server.rs)。
