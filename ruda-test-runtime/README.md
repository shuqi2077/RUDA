# ruda-test-runtime

Compile-time runtime selection for Ruda kernel tests. It exports a `TestRuntime` alias when `test-runtime` is enabled, so the same tests can target different drivers.

## Interfaces

- Enable `test-runtime` together with exactly one of `cpu`, `cuda`, `hip`, or `wgpu` to select that runtime.
- With `test-runtime` and no single backend selection, the build script selects WGPU.
- Without `test-runtime`, this crate does not export the `TestRuntime` alias.

## Usage

Cargo package: `ruda-test-runtime`. Rust import: `ruda_test_runtime`.

```toml
[dev-dependencies]
ruda-test-runtime = { version = "0.1", features = ["test-runtime", "cuda"] }
```

## Features

Default features: `std`, `ruda-kernel/frontend-default`, `ruda-kernel/library`, `ruda-driver-cpu?/default`, `ruda-driver-cuda?/default`, `ruda-driver-hip?/default`, `ruda-driver-wgpu?/default`.

| Feature | Purpose |
| --- | --- |
| `test-runtime` | Enable the TestRuntime alias and WGPU fallback dependency. |
| `cuda` | Select the CUDA test runtime. |
| `cpu` | Select the CPU test runtime. |
| `hip` | Select the HIP test runtime. |
| `wgpu` | Select the WGPU test runtime. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-test-runtime/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-test-runtime/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/programming-guide.md)
