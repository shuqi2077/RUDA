# ruda-io

Host-side download utilities used by Ruda data and model components. The `network` feature exposes a file downloader with a progress bar; the default `std` feature alone does not enable network downloads.

## Interfaces

- `network::downloader::download_file_as_bytes(url, message)` downloads a file into a byte vector.
- The downloader creates its own current-thread Tokio runtime and expects a response with a content length; request and read failures panic.

## Usage

Cargo package: `ruda-io`. Rust import: `ruda_io`.

```toml
[dependencies]
ruda-io = { version = "0.21", features = ["network"] }
```

## Features

Default features: `std`.

| Feature | Purpose |
| --- | --- |
| `std` | Enable standard-library mode. |
| `network` | Enable the Reqwest downloader, Tokio runtime, and progress reporting. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-io/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-io/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/model-inference.md)
