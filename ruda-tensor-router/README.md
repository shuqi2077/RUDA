# ruda-tensor-router

Routes framework tensor operations between multiple Ruda backends. Backend selection and transfer are represented by router tensors, clients, and channels.

## Interfaces

- `BackendRouter<R>` implements the backend over a runner channel.
- `Router<Backends>` uses `DirectByteChannel<Backends>` for a local backend tuple.
- `ByteBridge` transfers tensors through tensor data; it is not a zero-copy device-to-device bridge.

## Usage

Cargo package: `ruda-tensor-router`. Rust import: `ruda_tensor_router`.

```toml
[dependencies]
ruda-tensor-router = "0.21"
```

## Features

Default features: `std`.

| Feature | Purpose |
| --- | --- |
| `std` | Enable standard-library integration. |
| `distributed` | Enable distributed graph routing support. |
| `tracing` | Enable routing tracing. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-tensor-router/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-tensor-router/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md)
