# Driver API 与后端

[文档首页](README.md) · [Runtime API](runtime-api.md) · [兼容性](compatibility.md) · [English](../en/driver-api.md)

Ruda 驱动 crate 将通用运行时契约连接到具体执行后端。本页描述 Rust 后端入口，不定义 CUDA Driver API 的同名替代接口。

## 1. 后端组织

| crate | 执行后端 |
| --- | --- |
| `ruda-driver-cuda` | NVIDIA CUDA 驱动与编译路径 |
| `ruda-driver-cpu` | CPU |
| `ruda-driver-wgpu` | WGPU |
| `ruda-driver-hip` | HIP |

## 2. NVIDIA 设备选择

`ruda_driver_cuda::CudaDevice` 使用公开字段 `index: usize` 选择设备，默认索引为 0。`CudaRuntime` 实现通用 `Runtime` trait。

已有示例通过以下表达式获取默认设备客户端：

```rust
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::dsl::Runtime;

let client = CudaRuntime::client(&CudaDevice::default());
```

完整运行示例见 [ptx-runtime](../../ruda-driver-cuda/examples/ptx_runtime.rs)。设备索引标识当前机器的枚举位置，不是跨机器或设备重排后稳定不变的身份标识。

## 3. 配置入口

`RuntimeOptions` 包含 `memory_config`，用于内存管理配置。`CudaCompiler` 和 `CudaComputeKernel` 是当前 CUDA C++ 编译链的类型别名；启用直接 PTX 不会使这些别名自动代表另一套公开 API。

编译路径由 `RUDA_CUDA_COMPILER` 选择，见[编译器指南](compiler-guide.md)。`install::cuda_path()`、`install::include_path()` 与 `install::cccl_include_path()` 提供工具包路径定位。

## 4. 外部依赖边界

直接生成 PTX 省去的是该 Kernel 的 CUDA C++／NVRTC 编译步骤，执行仍依赖 NVIDIA 驱动。当前 crate 的依赖清单也仍保留 NVRTC。

现有入口没有承诺接管任意外部 CUDA context、stream 或裸设备指针。跨语言接入需要逐项核对所有权、执行依赖和错误传播，不能从“支持 CUDA 后端”推导出完整 ABI 兼容。

## 5. 接入其他后端

后端实现通过 `Runtime` 关联设备、编译器和计算服务；上层通过 `ComputeClient` 使用公共契约。先确认后端的存储、编译错误、同步与能力查询语义，再接入领域库。

源码入口：[CUDA 导出](../../ruda-driver-cuda/src/lib.rs)、[设备类型](../../ruda-driver-cuda/src/device.rs)、[运行时实现](../../ruda-driver-cuda/src/runtime.rs)、[Runtime trait](../../ruda/src/runtime/backend.rs)。
