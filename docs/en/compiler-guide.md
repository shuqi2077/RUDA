# Compiler Guide

[Documentation](README.md) · [PTX reference](ptx.md) · [Programming guide](programming-guide.md) · [中文](../zh/compiler-guide.md) | [日本語](../ja/compiler-guide.md) | [Deutsch](../de/compiler-guide.md) | [Русский](../ru/compiler-guide.md)

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

Features determine which code is built; environment variables select a path at runtime. These are separate controls. See the [CUDA manifest](../../ruda-driver/cuda/Cargo.toml) and [compiler manifest](../../ruda-compiler/Cargo.toml).

## 3. Environment variables

| Variable | Behavior |
| --- | --- |
| `RUDA_CUDA_COMPILER` | Accepts `nvrtc` or `ptx`; defaults to nvrtc when unset |
| `RUDA_PTX_VERSION` | Direct PTX requires an explicit `major.minor` version |
| `CUDA_PATH` | CUDA Toolkit installation root |
| `RUDA_PTX_TEST_CACHE` | Cache directory override read only by the ptx-runtime example |

Unknown compiler values produce an error. Selecting `ptx` without enabling `direct-ptx` also produces an error rather than switching to NVRTC. See [compiler_backend.rs](../../ruda-driver/cuda/src/compiler_backend.rs).

## 4. Targets and caches

Standalone direct PTX compilation requires an explicit PTX version and SM target. PTX identifies the instruction set version; SM identifies the target architecture. They are not interchangeable.

The CUDA driver's direct PTX cache namespace includes the backend identifier, SM, and PTX version, and is separate from the NVRTC cache.

## 5. Compilation errors

Unsupported IR, argument metadata, or target conditions produce errors. Direct PTX reports unsupported operations rather than automatically switching compilation paths. See [Debugging](debugging.md).

`ruda-compiler` also contains WGSL, SPIR-V, and MLIR modules.

## 6. Triton frontend

The source-only `ruda-triton` workspace member compiles Triton TTIR through a SIMT description and Ruda IR to direct PTX. Build its compiler with `cargo build --locked --release -p ruda-triton`, then install `./ruda-triton/python` with pip. Its Python package declares Python 3.10+, Triton 3.8+ and PyTorch dependencies. Set `RUDA_TRITON_PTX` to the built `target/release/ruda-triton-ptx` executable (`.exe` on Windows), or put it on PATH.

Set `TRITON_DEFAULT_BACKEND=ruda` before importing Triton. `RUDA_TRITON_DEVICE=cuda` is the default: tensors use `torch.cuda` storage and its current stream. `RUDA_TRITON_DEVICE=ruda` uses `ruda_torch` PrivateUse1 storage and RUDA's current native stream, flushing pending RUDA work before launch. These modes do not make CUDA and RUDA pointers interchangeable.

This frontend accepts Triton, not Gluon, and unsupported TTIR operations raise lowering errors. It is separate from the Rust kernel DSL; selecting it does not imply compatibility with every Triton kernel. GX1 mode requires a separate GX1 library not included in this software workspace.

Source: [Python installation metadata](../../ruda-triton/python/pyproject.toml), [compiler stages](../../ruda-triton/python/ruda_triton/compiler.py), [device/stream adapter](../../ruda-triton/python/ruda_triton/driver.py).

## 7. Direct-PTX optimization boundaries

The direct backend automatically lowers eligible constant U32/U64 division and remainder, vector arithmetic/conversions, aligned memory operations and warp reductions. The [PTX reference](ptx.md) explains exact integer semantics, SM/PTX gates, alignment fallback and participation rules.

Cooperative row-layout exploration is an explicit API over caller-provided IR, not automatic operator recognition or runtime autotuning. Reduced precision is a separate opt-in. Fewer PTX instructions alone do not establish faster execution; check register pressure and device timings. The default `nvrtc` compiler choice is unchanged.
