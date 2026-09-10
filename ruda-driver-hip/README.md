# ruda-driver-hip

AMD ROCm HIP execution backend for Ruda kernels. It uses `ruda-hip-sys` for HIP runtime bindings and the Ruda C++ compiler backend for device code.

## Interfaces

- `AmdDevice` identifies the device; `HipRuntime` implements execution.
- `device`, `runtime`, and `execution` contain the HIP backend interfaces.
- Executing kernels requires a compatible AMD device and ROCm/HIP installation.

## Usage

Cargo package: `ruda-driver-hip`. Rust import: `ruda_driver_hip`.

```toml
[dependencies]
ruda-driver-hip = "0.1"
```

## Features

Default features: `std`, `ruda/runtime-default`, `ruda-core/std`, `ruda-kernel/frontend-default`.

| Feature | Purpose |
| --- | --- |
| `rocwmma` | Select the rocWMMA matrix compiler path. |
| `std` | Enable standard-library integration. |
| `tracing` | Enable runtime tracing. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-driver-hip/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-driver-hip/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/driver-api.md)
