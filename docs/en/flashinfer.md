# Native inference operators

The reference operator vocabulary is [FlashInfer v0.7.4](https://github.com/flashinfer-ai/flashinfer/tree/v0.7.4),
commit `77039051b0be921788dc2784ec41423b558878ec`. These are native Rust/RUDA
interfaces, not a drop-in Python package or bindings to FlashInfer CUDA kernels.
Numerical kernels use the RUDA compiler and device runtime.

## Features and entry points

- `ruDNN/tensor-inference`: `rudnn::inference`.
- `ruDNN/inference-cuda`: explicitly owned CUDA IPC groups for native peer kernels.
- `ruDNN/inference-autotune`: native RUDA kernel/graph tuning sessions and persistent replay selection.
- `ruBLAS/tensor-float-quant`: `rublas::tensor_float_quant`.
- `ruPRIM/tensor-topk`, `ruPRIM/tensor-bitpack`: exact selection and bit packing.
- `ruRAND/tensor`: `rurand::tensor::sampling`.
- `ruda-llm/native-inference`: `rullm::native`; explicit device inference building blocks.
- `ruda-tensor-device/native-inference`: the same native tensor/operator interfaces.

These opt-in features do not change default ruLLM generation or checkpoint layouts.

## Operator interfaces

| Area | Rust module / types |
| --- | --- |
| Dense, ragged, sparse and MLA attention | `attention::{AttentionPlan, PreparedAttention, PreparedMla}` |
| Device CSR paged attention and cache transfers | `page::{CsrPages, PreparedPagedAttention, PreparedQuantizedAppend}`, `paged::PagedPlan` |
| Shared-prefix / multi-level composition | `cascade::{AttentionState, CascadePlan, PreparedCascade, merge_into}` |
| QSA and MSA scoring / selection / attention | `sparse::{QsaPlan, PreparedQsa}`, `msa::{MsaPlan, PreparedMsa}` |
| Paged MQA scores and top-k | `indexer::PreparedMqaIndexer`, `sharded_indexer::PreparedIcp` |
| Normalization and training | `norm::{PreparedNorm, PreparedBroadcast}`, `training::{PreparedRmsTraining, PreparedCausalConv}` |
| RoPE, video QKV and AdaLN / projection / packing | `rope`, `video`, `pre_attention::{PreparedPreAttention, PreparedTensorFp8Pack}` |
| Dense / block-scaled linear and expert execution | `linear::{Matrix, PreparedLinear, PreparedSegmented, PreparedLora, ScaledFp8}`, `experts::{PreparedRoutes, PreparedExperts}` |
| GDN / GDP / KDA / GDN2 and paged state pools | `recurrence::{DeltaPlan, PreparedDelta}`, `recurrence_paged::PreparedPagedDelta` |
| Mamba / SSD and speculative state caching | `mamba::SsmPlan`, `selective::PreparedSelectiveUpdate` |
| ReplaySSM and prefix-state materialization | `replay::{PreparedReplay, PreparedMaterialize}` |
| Integer / FP8 recurrent state storage | `state_storage::{QuantizedState, StateFormat, Rounding}` |
| mHC forward / backward | `mhc::{MhcPlan, PreparedMhcPreMap}` |
| Explicit peer-memory operation | `peer_moe`, `sharded_indexer`, `peer_candidates` |
| Stable configuration values | `config::{SfLayout, ActivationType, RoutingMethodType}` |

`Matrix` accepts dense floating storage, raw FP8 with floating decoder scales,
and native `BlockFloatTensor` FP4/FP8 storage. Scale layouts are explicit. A global
**decoder** scale multiplies decoded values; invert an upstream encoder scale
before using it as a decoder. FP8 payloads use U8 storage, not integer arithmetic.

## Device-buffer contracts

Plans upload CPU-owned scheduling metadata only, or retain device-resident CSR
and selection tables. Numerical tensors are not copied to the CPU. F32/F16/BF16
coverage paths accumulate in FP32. A format or operator name is not a claim of
bit-identical PTX, identical stochastic streams, or vendor-kernel performance.

`prepare` allocates retained outputs/workspace. Unsafe replay and `*_into` methods
require exclusive destinations, correctly ordered execution on the bound queue,
and live input/storage owners. `PreparedPack` packs strided views; its explicit
`write_back` commits mutations to the original view. Functional cache convenience
methods instead return copy-on-write storage, preserving existing aliases.

ReplaySSM ring cursors and accepted-token trackers are read-only caller-owned
metadata. Prepared ReplaySSM decodes only selected requests, not an entire state
pool. Prefix materialization snapshots sources before destination writes and
preserves raw state/scale bits for a zero-length replay prefix. Intermediate
per-token SSM history is opt-in because its memory use is `O(T*H*D*N)`.

Prepared sparse pipelines expose score/select/attend stages. Call these in order,
or provide producer operations that populate the same retained device buffers.
Page/state destination IDs must be unique when an operation performs parallel
writes. Peer mappings and native synchronization objects must outlive all queued
operations; cross-device mapping addresses are not portable checkpoint data.

Sampling accepts explicit uniforms or native Philox state. Invalid rows return
validity zero and token ID -1. Nucleus boundary ties are retained as a group, so
this FP32 path need not match the legacy CPU/F64 sampler token-for-token.

`trace::Definition` builds metadata-only operator definitions with symbolic/constant axes, tensor/scalar descriptors and JSON export. `trace_apply` accepts an explicit emission callback and never serializes tensor values or kernel source. `tuning::Session` reuses the existing native RUDA tuner; constructing or reloading a session does not benchmark or launch a GPU.

`SamplingPlan::allocate_workspace`, `sample_into`, and `chain_speculative_sampling_into` retain filtering, token and acceptance-count destinations. `uniforms_into` fills caller-owned Philox output storage.

## NVFP4 attention checkpoint interoperability

`nvfp4_attention::{PreparedQuantize, PackedQkv, PreparedAttention}` implements
independent 128-token Q/KV padding, full-sequence K centering, block/global Q
centering, compact FP32 QK correction, permuted K tokens, 64-row E4M3 scale
swizzles, and transposed V. Packed fields use U8 raw E2M1/E4M3 bytes. Forward
supports GQA, optional LSE and caller-owned outputs; causal alignment uses the
padded physical lengths, while `unpadded_k_len` masks the logical tail. This
portable native implementation does not require SM120 instructions and does
not promise the vendor kernel's intermediate probability rounding or speed.

`trace::ApplyRegistry` provides typed, shape-constrained native operation
substitution with explicit registration, deterministic priority and fallback.
It replaces the operation itself, rather than only logging a trace; it does not
install Python decorators or mutate process-global package functions.
