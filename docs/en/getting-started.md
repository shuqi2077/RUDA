# Installation and Quickstart

[Documentation](README.md) · [Next: Programming guide](programming-guide.md) · [中文](../zh/getting-started.md)

## 1. Choose your entry point

| Task | Entry point |
| --- | --- |
| Write GPU kernels | `ruda-kernel::dsl` and a device runtime |
| Use matrix multiplication, FFTs, or reductions | [Compute libraries](libraries/README.md) |
| Work with tensors and frameworks | [Tensors and frameworks](tensor-framework.md) |
| Train models and save state | [Training guide](training.md) |
| Load models and generate text or process images | [Model inference guide](model-inference.md) |
| Integrate a device backend | [Driver API](driver-api.md) |

Use Ruda from the source workspace.

## 2. Prepare an NVIDIA environment

You need Rust/Cargo, a linker toolchain for your platform, an NVIDIA GPU driver, and the CUDA Toolkit. The CUDA backend includes NVRTC and toolkit build dependencies; enabling direct PTX does not remove them.

Check your environment from the source root:

```powershell
rustc --version --verbose
cargo --version
nvidia-smi
nvcc --version
cargo metadata --no-deps --format-version 1 --offline --locked
```

These commands do not compile Ruda. `--offline` requires the dependencies needed for resolution to be cached locally.

Set `CUDA_PATH` to select the CUDA Toolkit root. On Windows, point it to an installed version directory, not the parent containing multiple versions. See the [CUDA installation path interface](../../ruda-driver-cuda/src/lib.rs).

## 3. Build the example

With the build environment ready, run:

```powershell
cargo build --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

The `ptx-runtime` example requires `direct-ptx`. Enabling this feature alone does not change the default compiler.

## 4. Select a compilation path and run

Choose one path in a separate PowerShell session. Resolve any command failure before continuing.

Default CUDA C++/NVRTC path:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

Direct PTX path:

```powershell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

The PTX version must match your target GPU and driver. See the [PTX backend reference](ptx.md).

The example performs FP32 addition at several lengths, checks results and tail sentinels across repeated executions, and prints cache counters. See [Examples and tutorials](samples.md) for optional cases.

## 5. Continue developing

Follow device selection, data upload, and kernel launch in the example, then read the [programming guide](programming-guide.md). For build, driver loading, or execution errors, use [Debugging and diagnostics](debugging.md) to locate the failing stage.
