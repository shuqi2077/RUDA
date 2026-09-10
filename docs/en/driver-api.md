# Driver API and Backends

[Documentation](README.md) · [Runtime API](runtime-api.md) · [Compatibility](compatibility.md) · [中文](../zh/driver-api.md) | [日本語](../ja/driver-api.md) | [Deutsch](../de/driver-api.md) | [Русский](../ru/driver-api.md)

Ruda driver crates connect the general-purpose runtime contract to execution backends. This guide describes Rust backend entry points, not drop-in replacements for CUDA Driver API functions.

## 1. Backend crates

| Crate | Execution backend |
| --- | --- |
| `ruda-driver-cuda` | NVIDIA CUDA driver and compilation paths |
| `ruda-driver-cpu` | CPU |
| `ruda-driver-wgpu` | WGPU |
| `ruda-driver-hip` | HIP |

## 2. Select an NVIDIA device

`ruda_driver_cuda::CudaDevice` selects a device through its public `index: usize` field, defaulting to 0. `CudaRuntime` implements the `Runtime` trait.

Obtain the default device's client with:

```rust
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::dsl::Runtime;

let client = CudaRuntime::client(&CudaDevice::default());
```

See [ptx-runtime](../../ruda-driver-cuda/examples/ptx_runtime.rs) for a complete runnable example. A device index identifies the current machine's enumeration position, not a stable identity across machines or changes in enumeration order.

## 3. Configuration

`RuntimeOptions` contains `memory_config` for memory management. `CudaCompiler` and `CudaComputeKernel` are type aliases for the CUDA C++ compilation chain; enabling direct PTX does not change their meaning.

Select the compilation path with `RUDA_CUDA_COMPILER`, as described in the [compiler guide](compiler-guide.md). `install::cuda_path()`, `install::include_path()`, and `install::cccl_include_path()` locate toolkit paths.

## 4. External dependencies

Direct PTX generation bypasses the kernel's CUDA C++/NVRTC compilation step. Execution still requires the NVIDIA driver, and the crate retains NVRTC dependencies.

The interfaces do not guarantee adoption of arbitrary external CUDA contexts, streams, or raw device pointers. Cross-language integration must account for ownership, execution dependencies, and error propagation; a CUDA backend alone does not provide complete ABI compatibility.

## 5. Integrate another backend

A backend uses `Runtime` to associate a device, compiler, and compute server. Higher layers access the contract through `ComputeClient`. Establish storage, compilation error, synchronization, and capability-query semantics before integrating compute libraries.

Source entry points: [CUDA exports](../../ruda-driver-cuda/src/lib.rs), [device type](../../ruda-driver-cuda/src/device.rs), [runtime implementation](../../ruda-driver-cuda/src/runtime.rs), and [Runtime trait](../../ruda/src/runtime/backend.rs).
