# Ruda Programming Guide

[Documentation](README.md) · [Runtime API](runtime-api.md) · [Compute libraries](libraries/README.md) · [中文](../zh/programming-guide.md) | [日本語](../ja/programming-guide.md) | [Deutsch](../de/programming-guide.md) | [Русский](../ru/programming-guide.md)

## 1. Host and device

Host Rust code selects devices, prepares inputs, constructs arguments, and reads results. Device kernels describe parallel computation. The frontend expands them into IR for backend compilation and execution.

`ruda-kernel::dsl` is the general-purpose kernel frontend. Kernels use Rust syntax with the frontend's types, macros, and operations; arbitrary Rust programs and standard library code cannot simply be compiled to a GPU.

The tensor framework dispatches operations to compute libraries. Applications do not need to write thread-level kernels to use matrix multiplication.

## 2. Execution hierarchy

| Ruda concept | Purpose |
| --- | --- |
| `RudaCount` | Number of workgroups in a launch |
| `RudaDim` | Execution dimensions of each workgroup |
| `ABSOLUTE_POS` | Global position in a one-dimensional elementwise kernel |
| `Array<T>` | One-dimensional array access in kernels |
| `Tensor<T>` | Kernel tensor access with shape and stride metadata |
| `Runtime` | Associates compiler, compute server, and device types |

For corresponding CUDA concepts, see the [compatibility guide](compatibility.md). Public exports are in the [DSL prelude](../../ruda-kernel/src/dsl/prelude.rs).

## 3. Your first kernel

This kernel comes from the [ptx-runtime example](../../ruda-driver-cuda/examples/ptx_runtime.rs), which includes the complete host program and execution checks:

```rust
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
fn add(a: &Array<f32>, b: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}
```

Import the macro and types with `ruda_kernel::dsl::prelude::*`. The output length excludes excess tail threads; both input arrays must contain at least as many elements as the output.

The example launches 64 execution units per workgroup and rounds the workgroup count up. This is the example's configuration, not a universally optimal kernel size.

## 4. Memory and arguments

Use `ComputeClient` to create or allocate device buffers, then construct kernel arguments from their handles. Distinguish byte counts from element counts:

- `client.empty(size)` takes a size in bytes.
- In the example's `ArrayArg::from_raw_parts(handle, count)`, `count` is the number of array elements.
- `RudaTensor<R>` carries a storage handle, shape, strides, dtype, device, and quantization parameters.

Cloning a handle or tensor does not copy the underlying device data. To change a layout, use the appropriate contiguous conversion, copy, or transformation operation. Editing metadata alone does not rearrange storage.

## 5. Submission, readback, and synchronization

Kernel submission and result availability are separate stages. Completion of host submission does not establish device execution time or success.

`read_one` waits for readback and returns a `Result`; `read_async` provides asynchronous results. `sync()` returns a future that must be awaited. `flush()` submits queued commands; it does not replace reading the result.

Multiple streams accessing the same data must respect producer-consumer dependencies. `set_stream` is unsafe, and host variable lifetimes alone do not establish device task completion.

## 6. Safety boundaries

Types, ownership, and borrowing constrain host resources and interfaces. Low-level wrappers must also maintain device execution requirements:

- Argument storage ranges, dtype, alignment, and layout match kernel accesses.
- Data used by asynchronous tasks remains valid until completion.
- Shared writes across threads and streams are correctly synchronized.
- Callers constructing raw arguments or using unchecked launches satisfy their safety contracts.

A checked launch is not a complete safety proof for an arbitrary kernel. The example uses explicit `unsafe` blocks with safety explanations for raw argument construction and launch. See the [Runtime API](runtime-api.md).

## 7. From kernels to compute libraries

Use [ruBLAS](libraries/rublas.md), [ruDNN](libraries/rudnn.md), and [ruPRIM](libraries/ruprim.md) for common operations. When working at kernel level, specify the input layout, accumulation precision, and execution configuration.
