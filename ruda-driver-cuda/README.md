# ruda-driver-cuda

NVIDIA CUDA execution backend for the Ruda runtime. It connects kernel compilation, CUDA device resources, memory, and execution queues through `CudaRuntime`.

## Interfaces

- `CudaDevice` selects a CUDA device; `CudaRuntime` implements the runtime.
- `install::cuda_path`, `include_path`, and `cccl_include_path` locate CUDA installation files.
- CUDA C++/NVRTC is the default compiler path. Direct PTX is selected explicitly with `direct-ptx` and `RUDA_CUDA_COMPILER=ptx`.

## Usage

Cargo package: `ruda-driver-cuda`. Rust import: `ruda_driver_cuda`.

```toml
[dependencies]
ruda-driver-cuda = "0.1"
```

## Features

Default features: `std`, `ruda/runtime-default`, `ruda-core/std`, `ruda-kernel/frontend-default`.

| Feature | Purpose |
| --- | --- |
| `direct-ptx` | Enable the direct PTX compiler path. |
| `ptx-wmma` | Select the PTX WMMA compiler implementation. |
| `tracing` | Enable cross-layer execution tracing. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-driver-cuda/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-driver-cuda/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/driver-api.md)

## Stream and event interop

`ruda_driver_cuda::interop::{command, StreamCommand, record_allocation}` operates on the same device service and CUDA context as RUDA kernels. Stream/event IDs are RUDA-managed identifiers, not raw CUDA handles; stream 0 is the default stream.

- `command(device, StreamCommand::Create)` creates a stream ID. `Validate`, `Query` and `Synchronize` inspect or wait for it.
- `Record { stream, event: 0, timing }` creates and records an event; pass its returned ID to record it again. `Wait { stream, event }` inserts a GPU-side dependency without waiting for GPU completion on the host.
- `EventQuery` reports readiness; `EventSynchronize` waits; `EventDestroy` releases the event. `Elapsed { start, end }` requires two timing-enabled events and returns FP32 milliseconds encoded in a `u64`; decode with `f32::from_bits(value as u32)`.
- `DeviceSynchronize` waits for the context and reports deferred RUDA launch errors. Commands return `Result<u64, ServerError>`; readiness is encoded as 0 or 1.

`record_allocation(device, stream, handle)` retains an allocation until already-submitted work on that stream completes; it does not replace an execution dependency. Stream creation is bounded by `streaming.max_streams` and fails when the pool is exhausted. These APIs do not import arbitrary external CUDA streams or contexts.
