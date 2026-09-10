# Compatibility Guide

[Documentation](README.md) · [Programming guide](programming-guide.md) · [中文](../zh/compatibility.md)

## 1. CUDA concepts

| Familiar CUDA concept | Ruda entry point |
| --- | --- |
| Host/device responsibilities | Host Rust code and the kernel DSL |
| Grid/block | RudaCount/RudaDim |
| One-dimensional global thread position | ABSOLUTE_POS |
| Allocation and transfers | ComputeClient memory and readback interfaces |
| Kernel compilation and launch | ruda-kernel, ruda-compiler, and the device runtime |
| BLAS, DNN, FFT, and sparse libraries | ruBLAS, ruDNN, ruFFT, and ruSPARSE |
| Collective communication | ruCCL |

This maps concepts, not drop-in function replacements. Follow Ruda's API ownership, argument, synchronization, and error contracts.

## 2. CUDA C++ and PTX

Ruda provides the CUDA C++/NVRTC path and an explicitly selected direct PTX path. Both execute through the NVIDIA driver.

The CUDA C++ compilation path does not provide unchanged compilation of arbitrary CUDA projects, complete CUDA Runtime/Driver ABI replacement, or direct relinking of existing library binaries. There is no command in this guide to automatically convert all CUDA applications.

Direct PTX handles the Kernel IR implemented by the generator, not arbitrary PTX input programs. Unsupported operations do not automatically fall back to another compiler.

## 3. Backends and dtypes

Backends differ in scalar types, atomics, matrix instructions, memory layouts, and synchronization. Query device capabilities, then check each operation's type and layout requirements.

A type in the shared DType enumeration is not necessarily available for every operation on every backend. The same generic Rust interface does not guarantee identical rounding or performance.

HIP uses a separate execution interface; do not apply PTX instruction set version numbers to it.

## 4. Cargo and naming

Library display names, Cargo package names, and Rust import names may differ. See the [compute library index](libraries/README.md). Features control available combinations; a default build does not enable every path.

The kernel frontend uses `#[ruda]`, RudaCount, and RudaDim.

## 5. Versions and numerical validation

The source version, lockfile, features, compiler backend, PTX/SM, driver, and GPU together define a validation configuration.

When replacing a compute library call, match layout, transpose, index base, input and accumulation dtypes, normalization, special values, and synchronization. For example, ruFFT pads non-power-of-two lengths; it is not a same-semantics replacement for an arbitrary-length FFT.
