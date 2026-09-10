# ruda-driver-wgpu

WGPU execution backend for Ruda kernels. It provides device initialization, storage, runtime options, and compiler selection for supported graphics APIs.

## Interfaces

- `WgpuDevice`, `WgpuRuntime`, and `WgpuStorage` provide the backend entry points.
- `init_setup`, `init_setup_async`, and `init_device` initialize or integrate device resources.
- `WgslCompiler` handles WGSL; additional compiler paths are feature-selected.

## Usage

Cargo package: `ruda-driver-wgpu`. Rust import: `ruda_driver_wgpu`.

```toml
[dependencies]
ruda-driver-wgpu = "0.1"
```

## Features

Default features: `ruda/runtime-default`, `ruda-core/std`, `ruda-kernel/frontend-default`.

| Feature | Purpose |
| --- | --- |
| `spirv` | Enable the Vulkan SPIR-V compiler path. |
| `msl` | Enable Metal shader compilation on supported Apple targets. |
| `vulkan-validate` | Enable Vulkan validation. |
| `profile-tracy` | Enable Tracy profiling integration. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-driver-wgpu/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-driver-wgpu/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/driver-api.md)
