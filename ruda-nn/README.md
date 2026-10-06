# ruda-nn

Neural-network layers, activation modules, padding, and loss functions for Ruda models. Layers operate through tensor backend contracts rather than selecting a GPU driver themselves.

## Interfaces

- `modules` contains neural-network layer implementations and is re-exported at the crate root.
- `activation` exposes activation modules; `loss` exposes loss functions.
- `Initializer` is re-exported from `ruda-model` for parameter initialization.
- `LoRALinearConfig::init(base)` freezes an existing dense projection and adds trainable A/B adapters. `merge()` produces a frozen, dropout-free dense projection.
- `loss::CausalLanguageModel` separates decoder hidden states from the vocabulary head. `CausalCrossEntropyConfig` projects full-vocabulary token chunks, handles shifted/ignored labels, and returns an FP32 loss sum with the effective token count. Chunking does not recompute the backward graph.

## Usage

Cargo package: `ruda-nn`. Rust import: `ruda_nn`.

```toml
[dependencies]
ruda-nn = "0.21"
```

## Features

Default features: `std`, `ruda-model/default`.

| Feature | Purpose |
| --- | --- |
| `std` | Enable standard-library model integration. |
| `sparse` | Enable sparse layer support. |
| `cuda` | Enable the CUDA tensor dependency. |
| `fusion` | Enable device-fusion integration. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-nn/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-nn/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md)
- [LoRA configuration, forward and merge](../docs/en/finetuning.md#rust-lora)
