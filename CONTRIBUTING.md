# Contributing to Ruda

**English** | [简体中文](docs/zh/CONTRIBUTING.md)

## Scope

Contributions cover the compute software stack: compilers, runtimes, kernels, compute libraries, tensors, model integration, documentation, and tests.

Keep library responsibilities separate rather than duplicating compute library operations in the tensor framework. Refactoring must not remove functionality or change dtype, operation order, error handling, or resource lifetimes. Report unsupported paths accurately rather than silently switching to the CPU, another compiler backend, or lower precision.

## Reporting issues

Provide the source version, operating system, Rust toolchain, GPU/driver/toolkit versions, enabled features, reproduction commands, minimal inputs, and actual/expected results. Remove tokens, personal paths, and models or data not authorized for disclosure before attaching logs. Do not submit model weights or build caches.

## Submitting changes

- Check component ownership and existing tests before editing. Explain dependencies and call contracts for cross-layer changes.
- Preserve third-party authorship, copyright, licenses, and provenance. Do not overwrite every file's license declaration with one license.
- Include tests for the behavior. Distinguish test source, syntax checks, successful compilation, and execution on actual devices.
- State which checks were not run. Cargo metadata and formatting checks are not compilation tests.
- For performance changes, report correctness and before/after measurements with the same inputs, dtype, device, and configuration. Do not substitute a simulator or reduced configuration for target hardware validation.

## Local checks

Run from the workspace root. These commands read manifests without compiling or running kernels:

```powershell
cargo metadata --no-deps --format-version 1 --offline --locked
cargo tree -p ruda-driver-cuda --no-default-features --features direct-ptx --edges normal,build --offline --locked
```

`--offline` requires the dependencies needed for resolution to be cached locally.

See [Getting started](docs/en/getting-started.md) for the build and demonstration entry point.

Contributors must have the right to submit their code. Original Ruda contributions use Apache-2.0; third-party code retains its original license and applicable terms.
