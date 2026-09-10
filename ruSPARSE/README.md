# ruSPARSE

**English** | [简体中文](docs/zh/README.md)

This repository is a source mirror. Run the commands below from the [RUDA monorepo](https://github.com/shuqi2077/RUDA) root.

Ruda's sparse-computing library, reusing shared tensors, runtimes, and device backends.

## Features and entry points

| Feature | Entry points and dependencies |
| --- | --- |
| Default (empty) | Sparse formats, validation, and host conversions; does not include device implementations |
| `tensor` | `CsrTensor<R>`, SpMV, SpMM, SpVV, SDDMM, SpGEAM, and SpGEMM; callers select the shared runtime |
| `cuda` | `tensor` plus the NVIDIA driver, for CUDA programs and examples |
| `cuda-tests` | Compatibility entry point for `cuda` tests |

## CUDA example

```sh
cargo run -p ruSPARSE --features cuda --example csrmv
```

The example computes `alpha * A * x + beta * y`, reads the result back synchronously, and checks it against `[7.0, 2.0, 18.5]`.

## Current computation contract

- The shared device path uses FP32 values and U32 indices; numerical kernels require 32-lane subgroups.
- Upload validated structures with `CsrTensor::from_csr`; `with_values` updates values, and `transpose` rearranges current device values. Operations return new results rather than overwriting inputs.
- SpGEAM/SpGEMM share host-side symbolic algorithms and preserve explicit-zero structure; the numerical stage runs on the device. Structural indices retain host copies; values are not read back for host-side computation.
