# ruda-store

Model and tensor serialization for Ruda. This package stores module snapshots, maps tensor names, filters paths, and reads or writes the selected checkpoint formats.

## Interfaces

- `ModuleSnapshot` and `ModuleStore` connect modules to storage.
- `SafetensorsStore`, `PytorchStore`, and `RudapackStore` are format-specific entry points.
- `PathFilter`, `KeyRemapper`, and module adapters support selective loading and cross-framework names.

## Usage

Cargo package: `ruda-store`. Rust import: `ruda_store`.

```toml
[dependencies]
ruda-store = "0.21"
```

## Features

Default features: `std`, `pytorch`, `safetensors`, `rudapack`, `memmap`.

| Feature | Purpose |
| --- | --- |
| `safetensors` | Enable SafeTensors storage. |
| `pytorch` | Enable PyTorch checkpoint loading. |
| `rudapack` | Enable Ruda's native storage format. |
| `memmap` | Enable memory-mapped file loading. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-store/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-store/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md)
