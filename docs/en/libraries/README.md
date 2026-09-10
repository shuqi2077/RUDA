# Compute Library Reference

[Documentation](../README.md) · [Tensors and frameworks](../tensor-framework.md) · [中文](../../zh/libraries/README.md)

Compute libraries implement operations, the runtime executes them on devices, and the tensor framework composes them. Library names describe responsibilities, not identical APIs or complete feature parity with similarly named CUDA libraries.

## Choose a library

| Library | Cargo package | Rust crate | Main operations |
| --- | --- | --- | --- |
| [ruBLAS](rublas.md) | `rublas` | `rublas` | Matrix multiplication, vector operations, grouped matrix multiplication, and INT4 |
| [ruDNN](rudnn.md) | `ruDNN` | `rudnn` | Attention, convolution, pooling, and MoE |
| [ruTENSOR](rutensor.md) | `ruTENSOR` | `rutensor` | General tensor contractions, einsum, reductions, permutations, and elementwise operations |
| [ruPRIM](ruprim.md) | `ruPRIM` | `ruprim` | Reductions, scans, elementwise operations, and indexing |
| [ruFFT](rufft.md) | `ruFFT` | `rufft` | Real FFT and inverse transforms |
| [ruRAND](rurand.md) | `ruRAND` | `rurand` | Uniform, normal, and Bernoulli distributions |
| [ruSPARSE](rusparse.md) | `ruSPARSE` | `rusparse` | Sparse matrix formats and operations |
| [ruCCL](ruccl.md) | `ruCCL` | `ruccl` | Collective communication and orchestration |

## Interface layers

- Kernel/launch interfaces accept device bindings and execution configuration for kernel and library development.
- Tensor interfaces handle allocation, layouts, and calls, commonly using `RudaTensor<R>`.
- Framework interfaces dispatch to libraries through components such as `ruda-tensor-device`.

Operations with the same name at different layers may have different arguments, return types, and error handling. Check the package, module path, and feature together; kernel entry point signatures are not tensor entry point signatures.

## Using the guides

Each library guide covers its purpose, features, interfaces, data contracts, and constraints. Explicitly select the features needed by a general-purpose path rather than assuming defaults suit every device.
