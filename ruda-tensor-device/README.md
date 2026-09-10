# ruda-tensor-device

Runtime-generic device tensor backend and dispatch into the Ruda domain libraries. It connects the framework's backend contracts to kernel-level `RudaTensor` values.

## Interfaces

- `DeviceBackend` and `DeviceRuntime` define the device backend integration.
- `dispatch` routes tensor operations to device kernels and domain libraries.
- `cuda::Cuda` supplies the NVIDIA tensor adapter when `cuda` is enabled.

## Usage

Cargo package: `ruda-tensor-device`. Rust import: `ruda_tensor_device`.

```toml
[dependencies]
ruda-tensor-device = "0.21"
```

## Features

Default features: `autotune`, `std`, `fusion`, `ruda-kernel/frontend-default`, `ruda-fusion?/device-default`.

| Feature | Purpose |
| --- | --- |
| `cuda` | Enable the CUDA tensor adapter. |
| `cuda-default` | Enable CUDA with the default fusion and tuning configuration. |
| `fusion` | Enable operation-fusion integration. |
| `autotune` | Enable domain-library autotuning. |
| `sparse` | Enable sparse tensor dispatch. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-tensor-device/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-tensor-device/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md)
