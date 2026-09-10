# ruda-tensor-wgpu

Framework tensor adapter over `ruda-driver-wgpu`. It exposes a `Wgpu` backend with optional fusion and graphics-API-specific compiler selections.

## Interfaces

- `Wgpu<F, I, B>` selects floating-point, integer, and Boolean element types.
- `WgpuDevice`, runtime options, and initialization helpers are re-exported.
- With `fusion`, `Wgpu` wraps the device backend in `ruda_fusion::Fusion`; without it, the alias names the device backend directly.

## Usage

Cargo package: `ruda-tensor-wgpu`. Rust import: `ruda_tensor_wgpu`.

```toml
[dependencies]
ruda-tensor-wgpu = "0.21"
```

## Features

Default features: `std`, `autotune`, `fusion`, `ruda-tensor-device/default`, `ruda-core/id-tests-std`, `ruda-test-runtime/default`.

| Feature | Purpose |
| --- | --- |
| `fusion` | Enable tensor-operation fusion. |
| `vulkan` | Enable the Vulkan/SPIR-V adapter. |
| `webgpu` | Enable the WebGPU/WGSL adapter. |
| `metal` | Enable the Metal/MSL adapter. |
| `template` | Enable source-template kernel integration. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-tensor-wgpu/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-tensor-wgpu/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md)
