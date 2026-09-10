# Runtime API Reference

[Documentation](README.md) · [Programming guide](programming-guide.md) · [Driver API](driver-api.md) · [中文](../zh/runtime-api.md) | [日本語](../ja/runtime-api.md) | [Deutsch](../de/runtime-api.md) | [Русский](../ru/runtime-api.md)

The general-purpose runtime is in `ruda::runtime`, enabled by the `ruda/runtime` feature. Each backend also requires its corresponding driver crate.

## 1. Core types

| Type | Responsibility | Definition |
| --- | --- | --- |
| `Runtime` | Associates Compiler, Server, and Device; provides device clients | [backend.rs](../../ruda/src/runtime/backend.rs) |
| `ComputeClient<R>` | Allocation, kernel submission, readback, synchronization, and capability queries | [client.rs](../../ruda/src/runtime/client.rs) |
| `ComputeServer` | Backend execution contract | [server module](../../ruda/src/runtime/server/mod.rs) |
| `RudaTensor<R>` | Device storage and tensor metadata | [Tensor definition](../../ruda-kernel/src/tensor/base.rs) |

`R::client(&device)` obtains the runtime's client. `R::Device` determines the device type; sharing a runtime type does not make storage on different physical devices interchangeable.

## 2. Memory and transfers

These methods belong to `ComputeClient<R>`:

| Method | Behavior |
| --- | --- |
| `create_from_slice(&[u8])` | Creates device data from host bytes and returns a Handle |
| `empty(usize)` | Allocates storage in bytes without guaranteeing zero initialization |
| `create_tensor_from_slice`, `empty_tensor` | Creates tensor storage layouts using shape and element size |
| `read_one(Handle)` | Synchronously reads one handle; returns `Result<Bytes, ServerError>` |
| `read_async(Vec<Handle>)` | Returns asynchronous readback results |
| `read(Vec<Handle>)` | Synchronously reads multiple handles; panics on error |
| `memory_usage()` | Queries memory usage tracked by the runtime |

`read_one_unchecked` panics if readback fails; its name does not mean that it disables kernel bounds checks. For noncontiguous tensors, use tensor readback interfaces rather than interpreting raw bytes as contiguous elements.

## 3. Execution control

| Method | Behavior |
| --- | --- |
| `launch` | Submits a kernel in Checked mode |
| `launch_unchecked` | Unsafe interface; BoundsCheckMode controls the actual checking mode |
| `flush` | Submits queued commands and returns a `Result` |
| `sync` | Returns a future that waits for execution to complete |
| `set_stream` | Unsafely sets the client's StreamId |

`launch` does not return computed device results. Asynchronous compilation or execution errors may surface during readback or synchronization. Macro-generated launch interfaces also construct arguments; their complete contracts are not interchangeable with client method signatures.

## 4. Capabilities and profiling

`properties()` returns device properties, and `features()` returns the feature set. Query the appropriate capabilities before selecting dtypes, atomics, or matrix instructions. Use `enumerate_devices`, `enumerate_all_devices`, and the count methods for enumeration. `profile` provides runtime profiling; distinguish submission, execution, and transfer in measurements.

## 5. Errors and safety

Low-level unchecked launches require callers to exclude out-of-bounds accesses and nonterminating loops. Layouts, binding lengths, and cross-stream lifetimes must match the kernel. See [Debugging](debugging.md) for check configuration and the [Driver API](driver-api.md) for device integration.
