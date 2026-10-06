# Runtime API Reference

[Documentation](README.md) · [Programming guide](programming-guide.md) · [Driver API](driver-api.md) · [中文](../zh/runtime-api.md) | [日本語](../ja/runtime-api.md) | [Deutsch](../de/runtime-api.md) | [Русский](../ru/runtime-api.md)

The general-purpose runtime is in `ruda::runtime`, enabled by the `ruda/runtime` feature. Each backend also requires its corresponding driver crate.

## 1. Core types

| Type | Responsibility | Definition |
| --- | --- | --- |
| `Runtime` | Associates Compiler, Server, and Device; provides device clients | [backend.rs](../../ruda-runtime/src/runtime/backend.rs) |
| `ComputeClient<R>` | Allocation, kernel submission, readback, synchronization, and capability queries | [client.rs](../../ruda-runtime/src/runtime/client.rs) |
| `ComputeServer` | Backend execution contract | [server module](../../ruda-runtime/src/runtime/server/mod.rs) |
| `RudaTensor<R>` | Device storage and tensor metadata | [Tensor definition](../../ruda-kernel/src/tensor/base.rs) |

`R::client(&device)` obtains the runtime's client. `R::Device` determines the device type; sharing a runtime type does not make storage on different physical devices interchangeable.

`ComputeClient::init(device, server)` registers a new server and panics if that server type is already registered for the device. `load(device)` requires an initialized compatible server; neither is a replacement for ordinary `R::client(&device)` device initialization.

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

`read_tensor(Vec<CopyDescriptor>)` returns `Vec<Bytes>` and panics on failure; `read_tensor_async` returns a future with `Result<Vec<Bytes>, ServerError>`. Descriptors must use runtime-compatible layouts: check `Runtime::can_read_tensor` and make unsupported tensor layouts contiguous before readback. The client does not automatically relayout arbitrary views. `memory_usage()` returns `Result<MemoryUsage, ServerError>` with server allocator accounting, not host RSS or total physical GPU/peak memory.

## 3. Execution control

| Method | Behavior |
| --- | --- |
| `launch` | Submits a kernel in Checked mode |
| `launch_unchecked` | Unsafe interface; BoundsCheckMode controls the actual checking mode |
| `flush` | Submits queued commands and returns a `Result` |
| `sync` | Returns a future that waits for execution to complete |
| `set_stream` | Unsafely sets the client's StreamId |

`launch` does not return computed device results. Asynchronous compilation or execution errors may surface during readback or synchronization. Macro-generated launch interfaces also construct arguments; their complete contracts are not interchangeable with client method signatures.

Calling `flush()` is not a completion fence. Await or resolve the future returned by `sync()` for the client's resolved execution stream; merely creating the future does not wait. Unsafe stream changes do not establish dependencies between producer and consumer work.

| Queue method | Contract |
| --- | --- |
| `execution_stream()` | Resolve an explicitly assigned stream or the calling thread's current stream. |
| `same_execution_queue(&other)` | Compare device/server identity and currently resolved stream; not a completion check. |
| `fixed_execution_queue()` | Clone the client with the currently resolved stream fixed, without creating a stream or waiting. Useful for plans retaining scratch across calls. |

## 4. Capabilities and profiling

`properties()` returns device properties, and `features()` returns the feature set. Query the appropriate capabilities before selecting dtypes, atomics, or matrix instructions. Use `enumerate_devices`, `enumerate_all_devices`, and the count methods for enumeration. `profile` provides runtime profiling; distinguish submission, execution, and transfer in measurements.

`device_id()` returns the runtime device identity; `properties_fingerprint()` returns cached hardware/capability identity without a per-call driver probe. Shared selection uses these through `runtime_environment(&client)` and the [stack autotuning API](stack-autotuning.md#controller-and-cache-api). Set immutable deployment tags before its first use; changing tags does not reconfigure an existing controller or memoized environment.

## 5. Errors and safety

Low-level unchecked launches require callers to exclude out-of-bounds accesses and nonterminating loops. Layouts, binding lengths, and cross-stream lifetimes must match the kernel. See [Debugging](debugging.md) for check configuration and the [Driver API](driver-api.md) for device integration.
