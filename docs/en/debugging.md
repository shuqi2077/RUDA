# Debugging and Diagnostics

[Documentation](README.md) · [Compiler](compiler-guide.md) · [Runtime API](runtime-api.md) · [中文](../zh/debugging.md)

## 1. Locate the failing stage

| Stage or symptom | Check first |
| --- | --- |
| Cargo manifest or dependency resolution failure | Paths, features, lockfile, and dependency cache |
| Rust build failure | First compiler error, toolchain, and feature combination |
| CUDA Toolkit lookup failure | CUDA_PATH, installation directory, and headers |
| Driver or device initialization failure | Driver availability, device index, and dynamic library loading |
| Direct PTX configuration failure | direct-ptx feature, compiler selection, and PTX version |
| Direct PTX compilation error | Unsupported operation, target, or argument layout named in the error |
| Readback or synchronization failure | Earlier asynchronous submissions, input bindings, and device errors |
| Incorrect numerical results | Shape, strides, dtype, bounds, synchronization, and algorithm contracts |

Keep the first error and its context, not just the final build failure summary.

## 2. Compilation and cache logs

Runtime configuration is in `ruda::runtime::config`. `CompilationConfig` provides logger, cache, and check_mode. `CompilationLogLevel` serializes as disabled, basic, or full; full includes source-level compilation information.

The `ptx-runtime` example counts compilations and PTX disk cache hits. `RUDA_PTX_TEST_CACHE` is read only by that example, not automatically by every application.

See [compilation.rs](../../ruda/src/runtime/config/compilation.rs) and [ptx_runtime.rs](../../ruda-driver-cuda/examples/ptx_runtime.rs).

## 3. Bounds checking

`BoundsCheckMode` has three configurations:

| Configuration | Runtime behavior |
| --- | --- |
| auto | Regular launches use checks; explicit unchecked launches may skip them |
| enforce | Enforces checks for launches |
| validate | Regular launches retain checks; unchecked paths select validation mode |

Detection capabilities depend on the compiler and backend. These modes do not automatically detect every out-of-bounds access, race, or lifetime error. Do not disable checks merely to let a failing case continue.

## 4. Asynchronous errors

`ComputeClient::launch` does not return computed results. Observe completion through readback returning a `Result` or by awaiting synchronization. Distinguish argument validation, kernel compilation, and device execution failures.

## 5. Reproduce an issue

The [examples page](samples.md) lists shared memory limits, tensor, bitwise, and half-precision cases.

Include the source version, features, complete command, backend, PTX version, GPU/driver/toolkit, inputs, and actual outputs when reporting a problem. Remove sensitive information and follow the [contributing guide](CONTRIBUTING.md).
