# ruda-runtime

Portable host runtime for Ruda compute backends. This crate owns compute clients and servers, device-storage management, execution streams, kernel tasks, and autotuning; hardware-specific execution lives in the separate driver crates.

## Interfaces

- `runtime::client` and `runtime::server`: submit work and implement compute services.
- `runtime::storage`, `memory_management`, and `allocator`: manage backend allocations.
- `runtime::backend`, `compiler`, and `kernel`: implement runtime and compiled-kernel contracts.

## Usage

Cargo package: `ruda-runtime`. Rust import: `ruda_runtime`.

```toml
[dependencies]
ruda-runtime = "0.1"
```

## Features

Default features: `runtime-default`.

| Feature | Purpose |
| --- | --- |
| `runtime` | Expose the runtime modules. |
| `runtime-std` | Enable host configuration and standard-library integration. |
| `runtime-storage-bytes` | Enable byte-backed storage support. |
| `runtime-tracing` | Enable runtime tracing. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-runtime/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-runtime/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/runtime-api.md)
