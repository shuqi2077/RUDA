# Ruda Documentation

[Documentation index](../README.md) · [中文](../zh/README.md) | [日本語](../ja/README.md) | [Deutsch](../de/README.md) | [Русский](../ru/README.md)

From your first GPU kernel to compute libraries, tensor training, and local model inference. Start with your task, then explore the programming guides and API references.

## Start here

| Your task | Reading path |
| --- | --- |
| Run your first GPU kernel | [Installation and quickstart](getting-started.md) → [Programming guide](programming-guide.md) |
| Use matrix, sparse, or neural network operations | [Compute libraries](libraries/README.md) → [Tensors and frameworks](tensor-framework.md) |
| Train models, accumulate gradients, and save state | [Training and saving state](training.md) |
| Fine-tune a local model with LoRA or NF4 | [Fine-tuning and recovery](finetuning.md) |
| Load local models, generate text, or process images | [Model loading and inference](model-inference.md) |

## Getting started

- [Installation and quickstart](getting-started.md): source setup, backend selection, and your first example.
- [Examples and tutorials](samples.md): vector addition, tensors, shared memory, and half precision.

## Programming guides

- [Ruda programming guide](programming-guide.md): host and device code, execution hierarchy, memory, synchronization, and safety.
- [Tensors and frameworks](tensor-framework.md): device tensors, library dispatch, fusion, and automatic differentiation.
- [Tensor recipes](tensor-recipes.md): complete CPU examples for matrix multiplication, layouts, dtype selection, and gradients.
- [Backend selection and composition](backend-composition.md): CUDA/ROCm/WGPU selection, local routing, and remote execution.
- [Data pipelines and model storage](data-and-storage.md): batching samples, saving weights, and importing checkpoint formats.
- [Training and saving state](training.md): training steps, FP32 masters/accumulation, mixed-storage checkpoints, token-weighted replicas, differentiable collectives and learning-rate schedules.
- [Ranks, devices and distributed training](distributed-training.md): explicit rendezvous, device placement, collective order, weighted gradients and rank-local recovery.
- [Architecture components and Python Muon](architecture-training.md): mHC, DSA/CSA/HCA, compressed KV caches and hybrid model composition.
- [LoRA and NF4 fine-tuning](finetuning.md): exact target selection, streamed weights, causal supervision, token-weighted accumulation, adapters and restart checkpoints.
- [General PyTorch model compiler](model-compiler.md): AOT forward/backward, native partitions, options and cache ownership.
- [Fixed-address PyTorch subgraphs](static-pytorch-graphs.md): explicit GraphOp plans, output lifetime, scratch reuse and first-order training.
- [Model loading and inference](model-inference.md): ruLLM, text and image inputs, sampling, AWQ, and continuous batching.

## Compilation and execution

- [Compiler guide](compiler-guide.md): the Rust kernel frontend, IR, CUDA C++/NVRTC, and direct PTX.
- [PTX backend reference](ptx.md): target configuration, compilation output, constraints, and errors.
- [Shared stack autotuning](stack-autotuning.md): participating operators, offline calibration, timing, cache identity and policy parameters.

## API references

- [Full-stack crate and API index](api-reference.md): package names, Rust imports, responsibilities, and source entry points across the workspace.
- [Runtime API](runtime-api.md): device clients, memory, submission, readback, and synchronization.
- [Driver API and backends](driver-api.md): backend types, device selection, and runtime integration.
- [Native PyTorch API](native-pytorch-api.md): device/component versions, shape/dtype contracts, normalization, optimizers, streams, attention, sequence training and quantization.
- [Compute library reference](libraries/README.md): library selection, Cargo features, and entry points.

## Compute libraries

| Library | Guide |
| --- | --- |
| ruBLAS | [Linear algebra and grouped matrix multiplication](libraries/rublas.md) |
| ruDNN | [Neural network operations and MoE](libraries/rudnn.md) |
| ruPRIM | [Reductions, scans, and indexing](libraries/ruprim.md) |
| ruFFT | [Fast Fourier transforms](libraries/rufft.md) |
| ruRAND | [Random number generation](libraries/rurand.md) |
| ruSPARSE | [Sparse computation](libraries/rusparse.md) |
| ruCCL | [Collective communication](libraries/ruccl.md) |

## Debugging and compatibility

- [Debugging and diagnostics](debugging.md): compilation errors, asynchronous errors, caching, and numerical checks.
- [Compatibility guide](compatibility.md): CUDA concepts, backend differences, and API and compilation boundaries.
- [Contributing](CONTRIBUTING.md): reporting issues and development conventions.

For a first project, follow the quickstart, programming guide, and relevant library guide. Consult the compiler and API references when developing backends or kernels.

- [Muon and explicit Muon + AdamW groups (experimental)](muon.md)
