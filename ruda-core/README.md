# ruda-core

Shared types and contracts used across the Ruda runtime, compiler, and tensor libraries. This package provides device handles, stream identifiers, numeric types, tensor metadata, and feature-selected IR and compilation interfaces.

## Interfaces

- `device`, `stream_id`, and `reader`: shared device identity and data-readback interfaces.
- `tensor`: tensor metadata, shapes, element types, and host data.
- `ir`, `compiler`, `kernel`, and `launch`: compiler-facing representations and launch contracts.

## Usage

Cargo package: `ruda-core`. Rust import: `ruda_core`.

```toml
[dependencies]
ruda-core = "0.1"
```

## Features

Default features: `std`.

| Feature | Purpose |
| --- | --- |
| `tensor` | Expose tensor types. |
| `tensor-host-data` | Enable host tensor data and element support. |
| `ir` | Expose the intermediate representation. |
| `compiler` | Expose compiler and kernel contracts. |
| `compilation-cache` | Enable persistent compilation-artifact caching. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-core/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-core/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/compiler-guide.md)
