# ruda-kernel

Rust kernel DSL, reusable kernel building blocks, and runtime-generic device tensor operations. This is the kernel-authoring layer, not the high-level `ruda-tensor::api::Tensor` interface.

## Interfaces

- `dsl`: kernel types, IR expansion, and launch interfaces.
- `library`: reusable device-kernel building blocks.
- `tensor`: device tensor allocation, transfer, readback, and operations.
- `template`: host source templates integrated with compiled kernel tasks.

## Usage

Cargo package: `ruda-kernel`. Rust import: `ruda_kernel`.

```toml
[dependencies]
ruda-kernel = "0.1"
```

## Features

Default features: `frontend-default`.

| Feature | Purpose |
| --- | --- |
| `frontend` | Enable the kernel DSL. |
| `library` | Enable reusable kernels. |
| `device-tensor` | Enable device tensor operations. |
| `source-template` | Enable source-template kernels. |
| `lowering-cpp` | Enable C++ lowering. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-kernel/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-kernel/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/programming-guide.md)
