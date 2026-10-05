# Full-stack crate and API index

[Documentation](README.md) · [中文](../zh/api-reference.md)

## Start at the right layer

| Task | First API | Guide |
| --- | --- | --- |
| Write a GPU kernel | `ruda`, kernel DSL, runtime client | [Programming guide](programming-guide.md) |
| Use typed tensors and gradients | `ruda_tensor::api::Tensor`, `Autodiff<B>` | [Tensor recipes](tensor-recipes.md) |
| Choose a tensor execution target | `Host`, `Cuda`, `Rocm`, `Wgpu`, Router, Remote | [Backend composition](backend-composition.md) |
| Call a domain operator | ruBLAS, ruDNN, ruPRIM, and other domain crates | [Compute libraries](libraries/README.md) |
| Build and train a model | `Module`, `ruda-nn`, `ruda-optim` | [Training](training.md) |
| Train with half storage and FP32 updates | `Module::to_dtype`, `Fp32MasterOptimizer`, `GradientsAccumulator::accumulate_with_dtype` | [Mixed-precision training](training.md#fp32-masters-and-mixed-parameter-storage) |
| Preserve mixed storage and pending gradients | `TrainingRecord::capture_with_dtypes`, `restore_with_dtypes` | [Training state](training.md#save-and-restore-training-state) |
| Synchronize replicas or differentiate collectives | `DataParallel`, `ruda_autodiff::collective` | [Distributed training](training.md#replicated-training-and-differentiable-tensor-collectives) |
| Load samples or weights | `Dataset`, `DataLoaderBuilder`, `ModuleSnapshot` | [Data and storage](data-and-storage.md) |
| Run local model inference | `rullm` | [Model inference](model-inference.md) |

## Package names and source entry points

The Cargo package name, repository directory, and Rust import name can differ. For example, ruFFT lives in `ruFFT/`, is installed as `ruda-fft`, and is imported as `rufft`; ruSOLVER is installed as `ruda-solver` and imported as `rusolver`.

The tables cover the resolved workspace, including the path-dependent CANN driver and the test-support packages. Package links open their manifests; source links open the crate entry point. Features and versions are specified by each manifest, not by the directory name. Test fixtures and the native PyTorch extension are not ordinary crates.io installation targets.

## Kernel, compiler, and runtime

| Cargo package | Rust import / source | Responsibility |
| --- | --- | --- |
| [`ruda`](../../ruda/Cargo.toml) | [`ruda`](../../ruda/src/lib.rs) | Kernel facade and runtime re-exports; start here for GPU kernels |
| [`ruda-core`](../../ruda-core/Cargo.toml) | [`ruda_core`](../../ruda-core/src/lib.rs) | Shared IR, tensor data, dtype, device, and memory contracts |
| [`ruda-compiler`](../../ruda-compiler/Cargo.toml) | [`ruda_compiler`](../../ruda-compiler/src/lib.rs) | Kernel frontends and target code generation |
| [`ruda-kernel`](../../ruda-kernel/Cargo.toml) | [`ruda_kernel`](../../ruda-kernel/src/lib.rs) | Rust kernel DSL, launch interfaces, and device tensor utilities |
| [`ruda-runtime`](../../ruda-runtime/Cargo.toml) | [`ruda_runtime`](../../ruda-runtime/src/lib.rs) | Compute clients/servers, storage, streams, and scheduling |
| [`ruda-kernel-macros`](../../ruda-kernel-macros/Cargo.toml) | [`ruda_kernel_macros`](../../ruda-kernel-macros/src/lib.rs) | Procedural macros translating kernel Rust to IR |
| [`ruda-ir-macros`](../../ruda-ir-macros/Cargo.toml) | [`ruda_ir_macros`](../../ruda-ir-macros/src/lib.rs) | Derive helpers for IR implementations |

## Drivers and Ascend programs

| Cargo package | Rust import / source | Responsibility |
| --- | --- | --- |
| [`ruda-driver-cuda`](../../ruda-driver-cuda/Cargo.toml) | [`ruda_driver_cuda`](../../ruda-driver-cuda/src/lib.rs) | CUDA devices, allocation, compilation, and launch |
| [`ruda-driver-hip`](../../ruda-driver-hip/Cargo.toml) | [`ruda_driver_hip`](../../ruda-driver-hip/src/lib.rs) | AMD HIP runtime adapter |
| [`ruda-hip-sys`](../../ruda-hip-sys/Cargo.toml) | [`ruda_hip_sys`](../../ruda-hip-sys/src/lib.rs) | Low-level HIP runtime bindings |
| [`ruda-driver-wgpu`](../../ruda-driver-wgpu/Cargo.toml) | [`ruda_driver_wgpu`](../../ruda-driver-wgpu/src/lib.rs) | WGPU setup, storage, and compute execution |
| [`ruda-driver-cpu`](../../ruda-driver-cpu/Cargo.toml) | [`ruda_driver_cpu`](../../ruda-driver-cpu/src/lib.rs) | CPU kernel runtime driver, separate from Host tensor backend |
| [`ruda-driver-cann`](../../ruda-driver-cann/Cargo.toml) | [`ruda_driver_cann`](../../ruda-driver-cann/src/lib.rs) | Dynamically loaded CANN AscendCL interfaces |
| [`ruda-ascend-kernels`](../../ruda-ascend-kernels/Cargo.toml) | [`ruda_ascend_kernels`](../../ruda-ascend-kernels/src/lib.rs) | Rust Ascend device programs and instruction lowering |

## Compute libraries

| Cargo package | Rust import / source | Responsibility |
| --- | --- | --- |
| [`rublas`](../../ruBLAS/Cargo.toml) | [`rublas`](../../ruBLAS/src/lib.rs) | Matrix/vector operations and backend dispatch |
| [`ruDNN`](../../ruDNN/Cargo.toml) | [`rudnn`](../../ruDNN/src/lib.rs) | Attention, convolution, pooling, normalization, and MoE |
| [`ruPRIM`](../../ruPRIM/Cargo.toml) | [`ruprim`](../../ruPRIM/src/lib.rs) | Reductions, scans, indexing, and elementwise kernels |
| [`ruda-fft`](../../ruFFT/Cargo.toml) | [`rufft`](../../ruFFT/src/lib.rs) | Fourier transforms |
| [`ruRAND`](../../ruRAND/Cargo.toml) | [`rurand`](../../ruRAND/src/lib.rs) | Random sampling and distributions |
| [`ruSPARSE`](../../ruSPARSE/Cargo.toml) | [`rusparse`](../../ruSPARSE/src/lib.rs) | Sparse formats and operators |
| [`ruTENSOR`](../../ruTENSOR/Cargo.toml) | [`rutensor`](../../ruTENSOR/src/lib.rs) | Tensor contractions, permutations, and reductions |
| [`ruCCL`](../../ruCCL/Cargo.toml) | [`ruccl`](../../ruCCL/src/lib.rs) | Collective communication algorithms |
| [`ruda-solver`](../../ruSOLVER/Cargo.toml) | [`rusolver`](../../ruSOLVER/src/lib.rs) | Host scientific solvers and opt-in device solve kernels |
| [`ruintegrate`](../../ruINTEGRATE/Cargo.toml) | [`ruintegrate`](../../ruINTEGRATE/src/lib.rs) | Host quadrature, ODE integration, and event location |
| [`rublas-host`](../../ruBLAS/host/Cargo.toml) | [`rublas_host`](../../ruBLAS/host/src/lib.rs) | CPU strided and batched matrix multiplication |
| [`ruDNN-host`](../../ruDNN/host/Cargo.toml) | [`rudnn_host`](../../ruDNN/host/src/lib.rs) | CPU neural-network operator implementations |
| [`ruPRIM-host`](../../ruPRIM/host/Cargo.toml) | [`ruprim_host`](../../ruPRIM/host/src/lib.rs) | CPU tensor primitives and indexing |
| [`ruFFT-host`](../../ruFFT/host/Cargo.toml) | [`rufft_host`](../../ruFFT/host/src/lib.rs) | CPU real Fourier transforms |
| [`ruRAND-host`](../../ruRAND/host/Cargo.toml) | [`rurand_host`](../../ruRAND/host/src/lib.rs) | Host random-number generation |

## Tensors, differentiation, and execution composition

| Cargo package | Rust import / source | Responsibility |
| --- | --- | --- |
| [`ruda-tensor`](../../ruda-tensor/Cargo.toml) | [`ruda_tensor`](../../ruda-tensor/src/lib.rs) | Backend contracts and api::Tensor behind the api feature |
| [`ruda-tensor-config`](../../ruda-tensor-config/Cargo.toml) | [`ruda_tensor_config`](../../ruda-tensor-config/src/lib.rs) | Shared autodiff/fusion configuration |
| [`ruda-tensor-device`](../../ruda-tensor-device/Cargo.toml) | [`ruda_tensor_device`](../../ruda-tensor-device/src/lib.rs) | DeviceBackend, CUDA adapter, and domain-library dispatch |
| [`ruda-tensor-host`](../../ruda-tensor-host/Cargo.toml) | [`ruda_tensor_host`](../../ruda-tensor-host/src/lib.rs) | Host CPU tensor backend and strided layouts |
| [`ruda-tensor-wgpu`](../../ruda-tensor-wgpu/Cargo.toml) | [`ruda_tensor_wgpu`](../../ruda-tensor-wgpu/src/lib.rs) | WGPU tensor backend and device setup |
| [`ruda-tensor-rocm`](../../ruda-tensor-rocm/Cargo.toml) | [`ruda_tensor_rocm`](../../ruda-tensor-rocm/src/lib.rs) | ROCm tensor adapter |
| [`ruda-tensor-tch`](../../ruda-tensor-tch/Cargo.toml) | [`ruda_tensor_tch`](../../ruda-tensor-tch/src/lib.rs) | LibTorch tensor adapter |
| [`ruda-autodiff`](../../ruda-autodiff/Cargo.toml) | [`ruda_autodiff`](../../ruda-autodiff/src/lib.rs) | Gradient graph, backward pass, and recomputation |
| [`ruda-fusion`](../../ruda-fusion/Cargo.toml) | [`ruda_fusion`](../../ruda-fusion/src/lib.rs) | Tensor operation fusion planning and execution |
| [`ruda-tensor-router`](../../ruda-tensor-router/Cargo.toml) | [`ruda_tensor_router`](../../ruda-tensor-router/src/lib.rs) | Local multi-backend routing and byte bridges |
| [`ruda-tensor-remote`](../../ruda-tensor-remote/Cargo.toml) | [`ruda_tensor_remote`](../../ruda-tensor-remote/src/lib.rs) | Remote tensor client/server |
| [`ruda-communication`](../../ruda-communication/Cargo.toml) | [`ruda_communication`](../../ruda-communication/src/lib.rs) | Transport protocols, WebSocket, and tensor data service |

## Models, training, storage, and integrations

| Cargo package | Rust import / source | Responsibility |
| --- | --- | --- |
| [`ruda-model`](../../ruda-model/Cargo.toml) | [`ruda_model`](../../ruda-model/src/lib.rs) | Module parameters, configuration, records, and data loaders |
| [`ruda-model-macros`](../../ruda-model-macros/Cargo.toml) | [`ruda_model_macros`](../../ruda-model-macros/src/lib.rs) | Config, Module, and Record derive macros |
| [`ruda-model-codegen`](../../ruda-model-codegen/Cargo.toml) | [`ruda_model_codegen`](../../ruda-model-codegen/src/lib.rs) | Code generation used by the model derive macros |
| [`ruda-nn`](../../ruda-nn/Cargo.toml) | [`ruda_nn`](../../ruda-nn/src/lib.rs) | Neural-network layers, activations, and losses |
| [`ruda-optim`](../../ruda-optim/Cargo.toml) | [`ruda_optim`](../../ruda-optim/src/lib.rs) | Optimizers, gradient accumulation/clipping, and schedules |
| [`ruda-dataset`](../../ruda-dataset/Cargo.toml) | [`ruda_dataset`](../../ruda-dataset/src/lib.rs) | Indexed datasets, sources, and transforms |
| [`ruda-io`](../../ruda-io/Cargo.toml) | [`ruda_io`](../../ruda-io/src/lib.rs) | Host I/O and optional network downloads |
| [`ruda-store`](../../ruda-store/Cargo.toml) | [`ruda_store`](../../ruda-store/src/lib.rs) | Model snapshots, Rudapack, safetensors, and PyTorch import |
| [`ruda-llm`](../../ruLLM/Cargo.toml) | [`rullm`](../../ruLLM/src/lib.rs) | Model loading and autoregressive inference |
| [`ruda-torch-native`](../../ruda-torch/Cargo.toml) | [`ruda_torch_native`](../../ruda-torch/src/lib.rs) | Native cdylib for the Python ruda_torch package; source-build component |

## Test and consumer support

| Cargo package | Rust import / source | Responsibility |
| --- | --- | --- |
| [`ruda-test-runtime`](../../ruda-test-runtime/Cargo.toml) | [`ruda_test_runtime`](../../ruda-test-runtime/src/lib.rs) | Runtime/kernel test infrastructure |
| [`ruda-test-utils`](../../ruda-test-utils/Cargo.toml) | [`ruda_test_utils`](../../ruda-test-utils/src/lib.rs) | Kernel test helpers |
| [`ruda-facade-consumer`](../../ruda/tests/consumer/Cargo.toml) | [`ruda_facade_consumer`](../../ruda/tests/consumer/src/lib.rs) | External-consumer fixture for facade feature wiring |
| [`ruda-store-pytorch-tests`](../../ruda-store/pytorch-tests/Cargo.toml) | [`ruda_store_pytorch_tests`](../../ruda-store/pytorch-tests/src/lib.rs) | PyTorch format interoperability tests |
| [`ruda-store-safetensors-tests`](../../ruda-store/safetensors-tests/Cargo.toml) | [`ruda_store_safetensors_tests`](../../ruda-store/safetensors-tests/src/lib.rs) | Safetensors format interoperability tests |

## Find an individual method

Use the typed tensor methods under [ruda-tensor/src/api](../../ruda-tensor/src/api), backend traits under [ruda-tensor/src/backend](../../ruda-tensor/src/backend), and domain-specific modules linked from each library guide. Enable the feature that exposes the module before using its symbols.

For device memory, submission, and synchronization, read the [Runtime API](runtime-api.md). For backend-specific initialization and launch contracts, read the [Driver API](driver-api.md). For model parameter and record types, start with [ruda-model exports](../../ruda-model/src/lib.rs), rather than the compiler's similarly named IR types.
