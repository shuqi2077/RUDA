# ruda-tensor-tch

Ruda tensor backend implemented through the `tch` LibTorch bindings. It adapts LibTorch tensors and devices to Ruda's backend operation contracts.

## Interfaces

- `LibTorch` selects the backend and floating-point element type.
- `LibTorchDevice` identifies CPU, CUDA, MPS, or Vulkan devices.
- `TchTensor` and element conversions bridge LibTorch storage and Ruda types. Available execution devices depend on the installed LibTorch build.

## Usage

Cargo package: `ruda-tensor-tch`. Rust import: `ruda_tensor_tch`.

```toml
[dependencies]
ruda-tensor-tch = "0.21"
```

## Features

Default features: `std`.

| Feature | Purpose |
| --- | --- |
| `std` | Enable standard-library integration. |
| `doc` | Enable tch's documentation-only mode; this is not an execution backend. |
| `tracing` | Enable tensor-backend tracing. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-tensor-tch/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-tensor-tch/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md)
