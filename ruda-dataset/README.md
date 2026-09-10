# ruda-dataset

Dataset interfaces and data-source adapters for Ruda machine-learning pipelines. It separates indexed item access and transformations from the model layer's batching and loading.

## Interfaces

- `Dataset<I>` supplies `get`, `len`, `is_empty`, and iteration.
- `source` and `transform` provide data sources and transformations.
- `audio`, `vision`, `nlp`, and network download utilities are feature-selected.

## Usage

Cargo package: `ruda-dataset`. Rust import: `ruda_dataset`.

```toml
[dependencies]
ruda-dataset = "0.21"
```

## Features

Default features: `sqlite-bundled`.

| Feature | Purpose |
| --- | --- |
| `sqlite-bundled` | Enable the bundled SQLite dataset source. |
| `sqlite` | Enable SQLite using the system library. |
| `vision` | Enable vision dataset utilities. |
| `audio` | Enable audio dataset utilities. |
| `dataframe` | Enable Polars dataframe support. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-dataset/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-dataset/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md)
