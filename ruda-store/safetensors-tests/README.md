# ruda-store-safetensors-tests

Workspace integration tests for `ruda-store` SafeTensors model loading. This package groups the SafeTensors interoperability fixtures and tests; its library target is empty.

## Interfaces

- The integration suite includes multilayer model loading and shared backend setup.
- Tests use local Ruda storage and model components rather than exposing a separate serialization API.
- Use `ruda-store` in applications; run this test package from the RUDA workspace.

## Usage

```sh
cargo test --locked -p ruda-store-safetensors-tests --tests
```

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-store/safetensors-tests/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-store/safetensors-tests/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md)
