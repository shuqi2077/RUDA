# ruda-model-macros

Procedural derives for Ruda model configuration, module traversal, and records. This crate delegates token generation to `ruda-model-codegen`.

## Interfaces

- `#[derive(Module)]` generates module handling for parameters and submodules.
- `#[module(skip)]` excludes a field from module traversal and persistence.
- `#[derive(Config)]` and `#[derive(Record)]` generate configuration and record implementations.

## Usage

Cargo package: `ruda-model-macros`. Rust import: `ruda_model_macros`.

```toml
[dependencies]
ruda-model-macros = "0.21"
```

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-model-macros/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-model-macros/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md)
