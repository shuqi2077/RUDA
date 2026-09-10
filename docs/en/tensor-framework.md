# Tensors and Frameworks

[Documentation](README.md) · [Compute libraries](libraries/README.md) · [Programming guide](programming-guide.md) · [中文](../zh/tensor-framework.md) | [日本語](../ja/tensor-framework.md) | [Deutsch](../de/tensor-framework.md) | [Русский](../ru/tensor-framework.md)

## 1. Layers

| Layer | Component | Responsibility |
| --- | --- | --- |
| Shared data and contracts | ruda-core | Dtypes, shapes, devices, and compilation contracts |
| Device tensors | ruda-kernel::tensor | Storage, metadata, allocation, and layouts |
| Device Backend | ruda-tensor-device | Dispatches tensor operations to compute libraries |
| Tensor API | ruda-tensor | Backend-generic tensor interfaces |
| Fusion | ruda-fusion | Operation fusion |
| Automatic differentiation | ruda-autodiff | Automatic differentiation |
| Model and training components | ruda-model, ruda-nn, ruda-optim, ruda-store, ruda-dataset | Models, network modules, optimizers, storage, and data |

## 2. Device tensors

`RudaTensor<R>` contains client, handle, meta, device, dtype, and qparams. Storage handles are separate from shape/strides, and quantization parameters are stored separately.

Low-level operations must check that inputs share a device, dtypes match the computation, and quantized data carries the correct parameters. Contiguous storage, transposed views, and materialized copies are distinct.

See the [device tensor module](../../ruda-kernel/src/tensor/mod.rs) for allocation, contiguous conversion, reshape, permutation, transfer, and readback.

## 3. NVIDIA Backend

`ruda-tensor-device/cuda` enables `ruda_tensor_device::cuda`.

Without `cuda-fusion`, `Cuda<F, I>` aliases `DeviceBackend<CudaRuntime, F, I, u8>`. With that feature enabled, it uses a fusion wrapper. F defaults to f32 and I to i32. See [cuda.rs](../../ruda-tensor-device/src/cuda.rs).

This is a tensor Backend, not a CUDA Driver API handle. Check each operation's dtype and feature requirements when selecting a backend.

## 4. Compute library dispatch

Matrix operations go to ruBLAS, neural network operations to ruDNN, reductions and indexing to ruPRIM, FFTs to ruFFT, and random generation to ruRAND. See the device Backend's [dispatch modules](../../ruda-tensor-device/src/dispatch).

## 5. Sparse tensors, quantization, and batched readback

`ruda_tensor::api::CsrTensor<B>` combines sparse structure and floating-point value tensors through `SparseOps`. It provides sparse/dense multiplication, transpose, addition, gather, scatter-add, and sampled operations. It differs from `rusparse::tensor::CsrTensor<R>`: the former is generic over Backend, the latter over Runtime. See the [sparse guide](libraries/rusparse.md).

Quantization includes multidimensional block scales, packing along non-final axes and partial packs, layout transformations, selected indexing operations, and fused quantized readback. Logical shape differs from packed storage shape; FP8/FP4 encodings must not be converted as integer values. See [kernel quantization](../../ruda-kernel/src/quantization), [quantized tensor layouts](../../ruda-kernel/src/tensor/contiguous.rs), and [fusion transactions](../../ruda-fusion/src/ops/transaction.rs). Operation coverage differs by scheme.

Batched readback organizes descriptors by actual device and stream. See [tensor transactions](../../ruda-kernel/src/tensor/transaction.rs).

## 6. Training and model inference

- [Training and saving state](training.md): configure an autodiff Backend, update parameters, accumulate gradients, and save or restore training state.
- [Model loading and inference](model-inference.md): load local weights, construct chat prompts, generate with sampling, and process image inputs.
