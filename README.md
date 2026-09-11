# Ruda — Rust High-Performance Computing

![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange?logo=rust&logoColor=white)
![Language](https://img.shields.io/github/languages/top/shuqi2077/RUDA)
![Forks](https://img.shields.io/github/forks/shuqi2077/RUDA?style=flat)
![Issues](https://img.shields.io/github/issues/shuqi2077/RUDA)
![Last commit](https://img.shields.io/github/last-commit/shuqi2077/RUDA?display_timestamp=committer)

**English** | [简体中文](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/project.md) | [日本語](https://github.com/shuqi2077/RUDA/blob/main/docs/ja/project.md) | [Deutsch](https://github.com/shuqi2077/RUDA/blob/main/docs/de/project.md) | [Русский](https://github.com/shuqi2077/RUDA/blob/main/docs/ru/project.md)

Ruda is a Rust high-performance computing library, building a complete software stack from GPU kernels, compilers, and runtimes to mathematical computing, tensors, and models.

Ruda is building Rust compilation and execution paths targeting PTX, HIP, and custom ISAs, while retaining the CUDA C++ compilation path. Controlled low-level `unsafe` encapsulation, combined with Rust's type system, ownership, and borrowing at higher levels, balances low-level performance control with higher-level memory safety.

## Quick Start

Requires Git, Rust/Cargo, a linker toolchain, an NVIDIA GPU and driver, and the CUDA Toolkit. See [environment setup](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md) for installation details.

### Clone

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
```

### Run a GPU kernel

Select the direct PTX compiler in your shell:

```sh
# Bash
export RUDA_CUDA_COMPILER=ptx
export RUDA_PTX_VERSION=8.0
```

```powershell
# PowerShell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
```

Then build and run the example:

```sh
cargo run --release --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

The example runs FP32 addition on the GPU and prints `PASS` lines and compilation-cache counters. Select a [PTX version](https://github.com/shuqi2077/RUDA/blob/main/docs/en/ptx.md) supported by your GPU and driver.

### Generate text with ruLLM

Place a local Qwen3.5-0.8B model in `./models/qwen35`, or replace the path below with your model directory. Model files are not included; see [model setup](https://github.com/shuqi2077/RUDA/blob/main/docs/en/model-inference.md#prepare-a-local-model).

```sh
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "The capital of France is" 8 1
```

The example prints the generated text and token IDs. To use the CUDA C++ / NVRTC path instead, set `RUDA_CUDA_COMPILER` to `nvrtc` before running either example.

## Stack Organization

One repository, multiple crates with clearly defined responsibilities. From domain libraries to higher-level frameworks, the stack is organized in layers and developed together.

| Layer | Components |
| --- | --- |
| Shared contracts | `ruda-core` |
| Compilation and kernels | `ruda-compiler`, `ruda-kernel`, macro components |
| Runtime and driver backends | `ruda`, `ruda-driver-cuda/cpu/wgpu/hip` |
| Domain libraries | [ruBLAS](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rublas.md), [ruDNN](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rudnn.md), [ruPRIM](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/ruprim.md), [ruFFT](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rufft.md), [ruRAND](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rurand.md), [ruSPARSE](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rusparse.md) |
| Experimental numerical science | [ruSOLVER](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rusolver.md), [ruINTEGRATE](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/ruintegrate.md) |
| Collective communication | ruCCL, `ruda-communication` |
| Tensors and frameworks | `ruda-tensor*`, `ruda-autodiff`, `ruda-fusion` |
| Models and data | `ruda-model`, `ruda-nn`, `ruda-optim`, `ruda-store`, `ruda-dataset` |

## Experimental Numerical Science

`rusolver` adds host real/complex factorizations, SVD, sparse LU, row-partitioned CG and analytic pullbacks; opt-in FP32 batched LU/Cholesky/QR/eigen/CG device paths are separate. `ruintegrate` adds host quadrature, infinite-domain transforms, RK45, stiff BDF1 and event location. First-order host solver graph integration is opt-in via `ruda-autodiff/solver-host`. [Extended scope](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/science-extended.md).

See the guides for convergence and backend restrictions. The packages are workspace members but not default members.

## Paths to Hardware

- **NVIDIA GPUs:** CUDA C++ → NVRTC → PTX is the default compilation path. Direct IR → PTX generation is also available as an explicit choice. Both execute through the NVIDIA driver.
- **Additional execution backends:** Backend source is available for CPU, WGPU, and HIP. See [Compatibility](https://github.com/shuqi2077/RUDA/blob/main/docs/en/compatibility.md) for the scope of support.

## Explore and Contribute

- [Ruda documentation](https://github.com/shuqi2077/RUDA/blob/main/docs/en/README.md): Quickstart, programming guides, compilers, API references, and compute library manuals.
- [NVIDIA demo](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md): Explore the example and its requirements.
- [Contributing guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/CONTRIBUTING.md): Contribute to operators, compilers, runtimes, and frameworks.

If you care about Rust, GPU kernels, compilers, or high-performance computing, join us in taking this stack further and making it faster.

## Origins and Licensing

[Third-party notices](https://github.com/shuqi2077/RUDA/blob/main/THIRD_PARTY_NOTICES.md)

Original Ruda software code that the project has the right to license is available under the [Apache License 2.0](https://github.com/shuqi2077/RUDA/blob/main/LICENSE). Third-party files remain under their original licenses; the root license does not override the `MIT OR Apache-2.0` declarations in migrated components.
