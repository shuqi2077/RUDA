# ruda-tensor-rocm

Framework tensor adapter for AMD ROCm/HIP devices. It combines the generic device tensor backend with `HipRuntime`, optionally wrapped in operation fusion.

## Interfaces

- `Rocm<F, I, B>` is the tensor backend alias.
- `RocmDevice` re-exports the HIP driver's `AmdDevice`.
- Device execution requires the HIP runtime and a compatible AMD device; this package is not the raw HIP FFI layer.

## Usage

Cargo package: `ruda-tensor-rocm`. Rust import: `ruda_tensor_rocm`.

```toml
[dependencies]
ruda-tensor-rocm = "0.21"
```

## Features

Default features: `fusion`, `ruda-tensor-device/default`, `ruda-core/id-tests-std`, `ruda-test-runtime/default`.

| Feature | Purpose |
| --- | --- |
| `fusion` | Wrap the HIP device backend in tensor fusion. |
| `autotune` | Enable device-domain autotuning. |
| `autotune-checks` | Enable autotune checks. |
| `tracing` | Enable backend tracing. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-tensor-rocm/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-tensor-rocm/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md)
