# Compiler Guide

[Documentation](README.md) · [PTX reference](ptx.md) · [Programming guide](programming-guide.md) · [中文](../zh/compiler-guide.md)

## 1. Compilation pipeline

The general-purpose kernel path starts with the macros and types in `ruda-kernel::dsl`, produces Kernel IR, and uses a backend for lowering and code generation. `ruda-compiler` contains compiler implementations; device drivers pass their output to the execution environment.

NVIDIA offers two compilation paths:

| Selection | Kernel compilation pipeline |
| --- | --- |
| `nvrtc` (default) | Rust kernel frontend → IR → CUDA C++ → NVRTC → PTX |
| `ptx` | Rust kernel frontend → IR → PTX |

Both execute through the NVIDIA driver. The CUDA C++ compilation path is not an interface for importing arbitrary C++ projects or providing complete CUDA source compatibility.

## 2. Cargo features

| Component/feature | Purpose |
| --- | --- |
| `ruda-kernel/frontend` | Kernel DSL frontend |
| `ruda-kernel/lowering-cpp` | C++ lowering integration |
| `ruda-compiler/cpp` | C++ backend implementation |
| `ruda-compiler/ptx` | Direct PTX compiler |
| `ruda-driver-cuda/direct-ptx` | Enables direct PTX selection in the CUDA backend |

Features determine which code is built; environment variables select a path at runtime. These are separate controls. See the [CUDA manifest](../../ruda-driver-cuda/Cargo.toml) and [compiler manifest](../../ruda-compiler/Cargo.toml).

## 3. Environment variables

| Variable | Behavior |
| --- | --- |
| `RUDA_CUDA_COMPILER` | Accepts `nvrtc` or `ptx`; defaults to nvrtc when unset |
| `RUDA_PTX_VERSION` | Direct PTX requires an explicit `major.minor` version |
| `CUDA_PATH` | CUDA Toolkit installation root |
| `RUDA_PTX_TEST_CACHE` | Cache directory override read only by the ptx-runtime example |

Unknown compiler values produce an error. Selecting `ptx` without enabling `direct-ptx` also produces an error rather than switching to NVRTC. See [compiler_backend.rs](../../ruda-driver-cuda/src/compiler_backend.rs).

## 4. Targets and caches

Standalone direct PTX compilation requires an explicit PTX version and SM target. PTX identifies the instruction set version; SM identifies the target architecture. They are not interchangeable.

The CUDA driver's direct PTX cache namespace includes the backend identifier, SM, and PTX version, and is separate from the NVRTC cache.

## 5. Compilation errors

Unsupported IR, argument metadata, or target conditions produce errors. Direct PTX reports unsupported operations rather than automatically switching compilation paths. See [Debugging](debugging.md).

`ruda-compiler` also contains WGSL, SPIR-V, and MLIR modules.
