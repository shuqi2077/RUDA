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

## 6. CUDA stream and event interop

`ruda_driver_cuda::interop::{command, StreamCommand, record_allocation}` operates on the same device service and CUDA context as RUDA kernels. Stream/event IDs are RUDA-managed identifiers, not raw CUDA handles; stream 0 is the default stream.

- `command(device, StreamCommand::Create)` creates a stream ID. `Validate`, `Query` and `Synchronize` inspect or wait for it.
- `Record { stream, event: 0, timing }` creates and records an event; pass its returned ID to record it again. `Wait { stream, event }` inserts a GPU-side dependency without waiting for GPU completion on the host.
- `EventQuery` reports readiness; `EventSynchronize` waits; `EventDestroy` releases the event. `Elapsed { start, end }` requires two timing-enabled events and returns FP32 milliseconds encoded in a `u64`; decode with `f32::from_bits(value as u32)`.
- `DeviceSynchronize` waits for the context and reports deferred RUDA launch errors. Commands return `Result<u64, ServerError>`; readiness is encoded as 0 or 1.

`record_allocation(device, stream, handle)` retains an allocation until already-submitted work on that stream completes; it does not replace an execution dependency. Stream creation is bounded by `streaming.max_streams` and fails when the pool is exhausted. These APIs do not import arbitrary external CUDA streams or contexts.
