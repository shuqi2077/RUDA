# ruda-hip-sys

Low-level Rust FFI bindings to AMD HIP and HIPRTC, used by `ruda-driver-hip`. This package exposes the native runtime interface rather than a high-level tensor API.

## Interfaces

- The crate root re-exports generated HIP and HIPRTC bindings.
- `hipconfig` discovers HIP versions and library locations.
- The build script selects bindings from the installed HIP configuration. Do not manually enable the `hip_*` features; configure `HIP_PATH` or `ROCM_PATH` for the installation instead.

## Usage

Cargo package: `ruda-hip-sys`. Rust import: `ruda_hip_sys`.

```toml
[dependencies]
ruda-hip-sys = "=7.14.6085000"
```

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-hip-sys/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-hip-sys/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/driver-api.md)
