# ruda-store-pytorch-tests

Workspace integration tests for loading PyTorch checkpoints through `ruda-store`. The library target is empty; the useful content is the integration-test suite rather than an application API.

## Interfaces

- Tests cover layer records, tensor dtypes, nested modules, buffers, and key remapping.
- The suite uses Ruda model, neural-network, optimizer, and host-tensor packages through local development dependencies.
- Run the tests from the RUDA workspace so those path dependencies are available.

## Usage

```sh
cargo test --locked -p ruda-store-pytorch-tests --tests
```

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-store/pytorch-tests/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-store/pytorch-tests/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md)
