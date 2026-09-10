# ruda-tensor-remote

Remote tensor execution over Ruda communication protocols and backend routing. Client and server functionality can be selected independently.

## Interfaces

- `RemoteBackend` and `RemoteDevice` are available with `client`.
- `server` exposes server-side execution, including the WebSocket server entry points.
- Start a server for a concrete backend before connecting a `RemoteDevice`; the server owns execution on that backend's device.

## Usage

Cargo package: `ruda-tensor-remote`. Rust import: `ruda_tensor_remote`.

```toml
[dependencies]
ruda-tensor-remote = "0.21"
```

## Features

Default features: `client`, `server`.

| Feature | Purpose |
| --- | --- |
| `client` | Enable remote client types. |
| `server` | Enable remote execution servers. |
| `tracing` | Enable cross-layer remote execution tracing. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-tensor-remote/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-tensor-remote/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md)
