# ruda-communication

Communication contracts and transports shared by Ruda distributed components. This package handles protocol clients, servers, channels, and messages; tensor collective operations are provided by `ruCCL`.

## Interfaces

- `Protocol`, `ProtocolClient`, and `ProtocolServer` define communication interfaces.
- `Address`, `Message`, and `CommunicationChannel` represent endpoints and message transport.
- `websocket` implements WebSocket transport; `data_service` exposes tensor-data services.

## Usage

Cargo package: `ruda-communication`. Rust import: `ruda_communication`.

```toml
[dependencies]
ruda-communication = "0.21"
```

## Features

No features are enabled by default.

| Feature | Purpose |
| --- | --- |
| `websocket` | Enable WebSocket client/server transport. |
| `data-service` | Enable tensor-data service integration. |
| `tracing` | Enable communication-layer tracing. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-communication/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-communication/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md)
