# ruda-model-codegen

Token-generation implementation behind Ruda model derives. It accepts parsed `syn::DeriveInput` values and produces `proc_macro2::TokenStream` output for native Ruda model paths.

## Interfaces

- `derive_config` and `derive_config_native` generate configuration implementations.
- `derive_module` and `derive_module_native` generate module implementations.
- `derive_record` and `derive_record_native` generate record implementations.

## Usage

Cargo package: `ruda-model-codegen`. Rust import: `ruda_model_codegen`.

```toml
[dependencies]
ruda-model-codegen = "0.21"
```

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-model-codegen/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-model-codegen/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md)
