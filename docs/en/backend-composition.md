# Backend selection and composition

[Documentation](README.md) · [Tensor recipes](tensor-recipes.md) · [中文](../zh/backend-composition.md)

## Kernel runtime versus tensor backend

`cargo add ruda --features cuda` selects the CUDA kernel facade. It does not turn `ruda` into the typed tensor, model, and optimizer crates. For `Tensor<B, D>`, select a tensor backend and enable `ruda-tensor/api` (or `api-std`).

| Execution target | Tensor backend | Device | Selection |
| --- | --- | --- | --- |
| CPU | `ruda_tensor_host::Host` | `HostDevice` | `ruda-tensor-host`; `std`, optional `simd`/`rayon` |
| NVIDIA CUDA | `ruda_tensor_device::cuda::Cuda<f32, i32>` | `CudaDevice` | `ruda-tensor-device/cuda-default` |
| AMD ROCm | `ruda_tensor_rocm::Rocm<f32, i32>` | `RocmDevice` | `ruda-tensor-rocm` and ROCm/HIP runtime |
| WGPU | `ruda_tensor_wgpu::Wgpu<f32, i32>` | `WgpuDevice` | Select `vulkan`, `metal`, or `webgpu` for the intended target |
| LibTorch | `ruda_tensor_tch::LibTorch` | `LibTorchDevice` | Matching LibTorch installation |
| Multiple local backends | `ruda_tensor_router::Router<(B1, B2)>` | `duo::MultiDevice<B1, B2>` | Explicit device variant |
| Remote execution | `ruda_tensor_remote::RemoteBackend` | `RemoteDevice` | `client` feature and a running server |

Hardware drivers and target toolchains are still required. Enabling a Cargo feature does not install them or automatically provide all backends. The WGPU path selects a graphics compute API rather than automatically switching between native CUDA and HIP.

Ascend is exposed in this repository through `ruda-driver-cann`, `ruda-ascend-kernels`, the compiler's Ascend paths, and `rublas::cann`. These are separate from the typed tensor backend choices above; see the [compiler guide](compiler-guide.md) and [ruBLAS guide](libraries/rublas.md).

## CUDA tensor dependency

For typed tensor programs, add:

```toml
[dependencies]
ruda-tensor = { version = "0.21", default-features = false, features = ["api-std"] }
ruda-tensor-device = { version = "0.21", default-features = false, features = ["cuda-default"] }
```

Use `ruda_tensor_device::cuda::{Cuda, CudaDevice}` and choose `Cuda<f32, i32>` as the backend type. The full model/optimizer example is in [Training](training.md).

`cuda` exposes the adapter; `cuda-fusion` wraps it in `Fusion`; `cuda-default` combines CUDA fusion with default runtime/backend facilities. For ROCm and WGPU, their `fusion` features determine whether the exported backend alias includes the fusion wrapper. Cargo features are additive: another dependency enabling fusion can affect the final alias.

## Route tensors between CPU and WGPU

This example explicitly uses Vulkan, requiring a compatible Vulkan adapter. Create a binary project with:

```toml
[dependencies]
ruda-tensor = { version = "0.21", default-features = false, features = ["api-std"] }
ruda-tensor-host = { version = "0.21", default-features = false, features = ["std"] }
ruda-tensor-router = "0.21"
ruda-tensor-wgpu = { version = "0.21", features = ["vulkan"] }
```

Use this `src/main.rs` and run `cargo run`:

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

`B1` and `B2` follow tuple order. The router's default device selects the first backend; explicit variants make placement unambiguous. `trio` and `quad` provide corresponding three- and four-backend device enums.

`Router` uses `DirectByteChannel` and `ByteBridge`. A cross-backend transfer goes through `TensorData`; it is not a zero-copy or GPU peer-to-peer guarantee. `to_device` moves the tensor explicitly. The router does not automatically choose the fastest backend, split a model, or replace unsupported operations with a CPU implementation.

## Remote tensor execution

The server selects the actual backend; the client sends tensor operations over WebSocket. A CPU server is sufficient for learning the protocol. Use these dependencies in a new project:

```toml
[dependencies]
ruda-tensor = { version = "0.21", default-features = false, features = ["api-std"] }
ruda-tensor-host = { version = "0.21", default-features = false, features = ["std"] }
ruda-tensor-remote = { version = "0.21", default-features = false, features = ["client", "server"] }
```

Create `src/bin/server.rs`:

```rust
use ruda_tensor_host::{Host, HostDevice};

fn main() {
    ruda_tensor_remote::server::start_websocket::<Host>(HostDevice, 3000);
}
```

Create `src/bin/client.rs`:

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

Run `cargo run --bin server`, then run `cargo run --bin client` in a second terminal. The convenience server binds `0.0.0.0:3000`, not only loopback; its built-in transport does not configure authentication or TLS. The client URL above is for a same-machine connection.

`start_websocket` owns its Tokio runtime and blocks until the server stops. Inside an existing async runtime, use `start_websocket_async::<B>(device, port).await`. The server backend must implement `BackendIr`; replacing the server's backend also requires its dependencies and hardware. Remote execution is distinct from ruCCL distributed collectives.

## Choosing wrappers and boundaries

- Use `Autodiff<B>` when the tensor graph needs gradients; see [Tensor recipes](tensor-recipes.md).
- Fusion optimizes supported operation sequences; it does not add missing backend operations or hardware support.
- Router controls local placement and transfer; Remote places execution behind a transport.
- ruCCL provides collective communication, not the remote tensor server; see [ruCCL](libraries/ruccl.md).
- Backend APIs do not imply identical dtype, sparse, or quantized operator coverage. See [Compatibility](compatibility.md) and the relevant domain-library guide.

## API entry points

- [CUDA aliases](../../ruda-tensor-device/src/cuda.rs), [ROCm aliases](../../ruda-tensor-rocm/src/lib.rs), and [WGPU setup](../../ruda-tensor-wgpu/src/lib.rs).
- [Router aliases](../../ruda-tensor-router/src/lib.rs), [device enums and byte bridge](../../ruda-tensor-router/src/types.rs).
- [Remote client](../../ruda-tensor-remote/src/lib.rs), [server startup](../../ruda-tensor-remote/src/server/base.rs), and [WebSocket listener](../../ruda-communication/src/websocket/server.rs).
