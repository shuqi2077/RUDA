# Native PyTorch API reference

[Documentation](README.md) · [Package guide](../../ruda-torch/README.md) · [中文](../zh/native-pytorch-api.md)

## Device and component contracts

Import `ruda_torch` before creating tensors on `ruda:0`. Another PrivateUse1 backend cannot already be registered in the process. `is_available()` reports the initialized integration, not a complete inventory of GPU instruction support. `device_count()` is 1 and `current_device()` is 0. `synchronize(device=None)` accepts the native device only and waits for completion. `execution_stats()` returns cumulative native launch, transfer and selected-library counters; it is not a timer or an autotuning result.

Rust/C++ components require base ABI 10. Optional features negotiate their own versions: graph 3, training 4, router 1, sequence 1, NF4 decode 1 and NF4 matmul 1. Paged backward accepts native API 1 or 2 with a compatible bridge; ordered backward requires 2. A base-ABI match alone does not provide every optional feature. Use the matching [source build or precompiled bundle](../../ruda-torch/README.md#build-and-install).

The eager native backend does not install a generic CPU fallback. Unsupported operators, unavailable capabilities and device errors propagate. CUDA tensors and RUDA tensors are different allocations and are not interchangeable merely because they use the same GPU.

## Normalization and gated activation

The following fused functions require contiguous, dense FP32/FP16/BF16 `ruda:0` tensors without unresolved conjugate/negative views. Outputs retain the activation dtype; saved normalization statistics are FP32. Only first-order derivatives are supported.

| API | Parameters, shape and result |
| --- | --- |
| `rms_norm(x, weight=None, eps=None)` | Normalize the positive last dimension D of `[...,D]`. Optional weight `[D]` on the same device, input dtype or FP32. `eps=None` uses `torch.finfo(x.dtype).eps`; an explicit epsilon must remain finite and positive in FP32. |
| `RMSNorm(width, eps=1e-5, elementwise_affine=True, device=None, dtype=None)` | Positive width; optional affine parameter defaults to FP32. CPU construction is allowed, but move the module to RUDA before forward. `forward(x)` requires last-axis width equality. |
| `layer_norm(x, weight=None, bias=None, eps=1e-5)` | Last-axis normalization. Each supplied affine vector is `[D]`, input dtype or FP32, on the input device; positive finite FP32 epsilon. |
| `LayerNorm(width, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None)` | Positive width; affine storage defaults to FP32. `elementwise_affine=False` disables both parameters; `bias=False` disables bias alone. |
| `silu_mul(gate, up)` | Same shape/dtype/device, no broadcasting. Returns `SiLU(gate) * up` with the low-precision storage-rounding boundary retained. |

Do not mutate saved inputs/weights before backward. `create_graph=True` is rejected by these fused backward interfaces. Their optional FP32 affine parameters are a different contract from explicit StaticGraph RMSNorm, which requires same-dtype weights. Source: [training.py](../../ruda-torch/python/ruda_torch/training.py).

## Optimizer and loss scaling

`AdamW(params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-2, fused_step=False, max_grad_norm=None, hierarchical_stats=False)` requires nonoverlapping, normal leaf RUDA parameters with the same contiguous floating contract above. Sparse gradients and `amsgrad`, `maximize`, `differentiable`, `capturable` are not supported. LR/decay are nonnegative, epsilon positive and both betas in `[0,1)` after FP32 conversion.

- `step(closure=None, loss_scale=1.0)` updates active parameters using FP32 masters/moments. Clear gradients before the next accumulation window. The default path unscales gradients in place and reads a four-byte finite flag before updates.
- `fused_step=True` keeps gradients unchanged, reads a 12-byte statistics report and still uses one update kernel per active parameter. `max_grad_norm` is an optional nonnegative global unscaled L2 limit; it requires this mode.
- `hierarchical_stats=True` also requires fused mode and adds bounded reduction stages for large statistics workspaces.
- Nonfinite gradients skip the whole update. Inspect `last_step_skipped`, `last_step_had_grad`, and, for the fused path, `last_grad_norm`/`last_clip_coef`. This host-readback path is not optimizer graph capture.
- `state_dict()` / `load_state_dict()` preserve optimizer state and supported step options. Restore against matching parameter shapes and dtype contracts.

`GradScaler(init_scale=65536., growth_factor=2., backoff_factor=0.5, growth_interval=2000, min_scale=2**-24, max_scale=2**24)` is RUDA's scaler, not `torch.amp.GradScaler`. Its scale bounds are positive, growth factor greater than one, backoff in `(0,1)` and interval a positive integer. Use `scale(loss).backward()`, `step(optimizer)`, `update()` on one optimizer cycle; the loss must contain one FP32 element on RUDA. `step` accepts RUDA AdamW, Muon or MuonAdamW without a closure or extra arguments. There is no `unscale_()` or multi-optimizer cycle. `get_scale()` reads the current scale; snapshot scaler state only between completed cycles.

For Python Muon parameter grouping, Newton–Schulz options and checkpoint rules, see the [architecture and Python Muon guide](architecture-training.md#python-muon-and-explicit-adamw-exclusions) and [optimizer definitions](../../ruda-torch/python/ruda_torch/optim.py). The separate [Rust Muon guide](muon.md) describes Rust tensor optimizers. Do not interpret a local shard update as full-matrix Muon.

## Streams and events

| API | Execution/lifetime contract |
| --- | --- |
| `Stream(priority=0)` | Only priority zero is supported. Reuse streams; the underlying stream pool is bounded. |
| `current_stream(device=None)`, `default_stream(device=None)` | Native device only; the default stream has ID zero. |
| `stream(s)` | Context manager that restores the previous stream on exit. |
| `Stream.wait_stream(other)`, `wait_event(event)` | Insert device-side ordering; do not replace them with host readiness queries. |
| `Stream.record_event(event=None)` | Record and return an event on that stream. |
| `Event(enable_timing=False)` | `record(stream=None)` defaults to the current stream; `wait(stream=None)` inserts an event dependency. |
| `Stream.query()`, `Event.query()` | Report readiness without a host completion wait. |
| `Stream.synchronize()`, `Event.synchronize()` | Wait on the host. |
| `start.elapsed_time(end)` | Milliseconds; both events must have timing enabled and their work must be complete. |
| `record_stream(tensor, s)` | Retain allocation for already-submitted work; does not establish a producer/consumer dependency. |
| `Event.close()` | Release the native event explicitly; reuse requires recording again. |

Set `RUDA_TORCH_ASYNC=1` before first native submission to select asynchronous dispatch; it is cached for the process. Scalar extraction/readback and explicit synchronization still wait. See the [stream example](../../ruda-torch/README.md#streams-events-and-asynchronous-dispatch).

## Paged attention and fixed router weights

`PagedAttentionPlan(page_size=..., num_pages=..., block_tables=..., kv_lengths=..., sequence_ids=..., positions=..., splits=1, backward_strategy='atomic')` owns immutable scheduling metadata. Use matching contiguous same-dtype FP32/FP16/BF16 native tensors and execution queue; recreate the plan when its schedule changes.

| Method | Tensor contract |
| --- | --- |
| `attention(q, k, v, scale=..., causal=True)` | Q `[queries, query_heads, features]`; K/V `[physical_pages,page_size,kv_heads,features]` with their respective key/value feature widths. Query heads divisible by KV heads; finite positive scale, finite values, feature widths at most 1024. |
| `mla(absorbed_query, position_query, latent_cache, position_cache, scale=..., causal=True)` | Queries `[queries,heads,rank]` / `[queries,heads,position_dim]`; caches `[pages,page_size,1,rank]` / `[pages,page_size,1,position_dim]`; positional width at most 256. Returns `[queries,heads,rank]` context, before value/output projection. |
| `workspace_bytes(query_heads, value_dim)` | Planned forward split scratch only, not backward memory or total peak VRAM. |

`splits=1` is unsplit; values 2–32 use partial attention and FP32 merging with a 64 MiB per-workspace limit. Caller owns KV content/updates and positional encoding. Use the model's original MLA QK scale. First-order gradients are supported; arbitrary external masks, quantized KV storage and higher-order derivatives are not. Atomic backward is the default; `backward_strategy='ordered'` explicitly chooses atomic-free history reduction, not autotuning. Detailed ownership/compaction options: [package attention guide](../../ruda-torch/README.md#paged-gqa-and-mla).

`selected_router_weights(logits, indices, scoring='softmax', renormalize=False, scale=1.)` takes contiguous native logits `[tokens,experts]` in FP32/FP16/BF16 and indices `[tokens,top_k]` in int32/int64, with `1 <= top_k <= min(experts,64)`. It returns FP32 selected weights and first-order logits gradients, not expert selection. Softmax covers all experts before gather; sigmoid is pointwise. Optional renormalization covers gathered slots and scale is applied last. Duplicate indices accumulate gradients. Invalid index values yield NaN for the affected forward/backward row with bounds-safe reads rather than a host index synchronization. See [router implementation](../../ruda-torch/python/ruda_torch/_router.py).

## Sequence training and learned fake quantization

`solve_triangular(a, b, upper=False, left=True, unitriangular=False)` accepts same-device, same-floating-dtype matrices. A is `[...,N,N]`; B is `[...,N,K]` for a left solve or `[...,K,N]` for a right solve, with broadcastable batch dimensions. Flags must be Python bools. RUDA uses sequence API 1; CPU/CUDA are explicitly selected references. First-order autograd is provided.

`gated_delta_rule(query, key, value, beta, log_decay, initial_state=None, query_scale=None, normalize_qk=False, norm_eps=1e-6, chunk_size=64, checkpoint_chunks=True, native_forward=True)` requires Q/K `[B,H,T,Dk]`, V `[B,H,T,Dv]`, beta/decay `[B,H,T]`, optional initial state `[B,H,Dk,Dv]` on one floating device. Q/K/V storage dtypes match. `chunk_size` is positive. The result is `(output, final_state)`; output is `[B,H,T,Dv]`. State/intermediates use FP32, or FP64 for explicit CPU FP64 input. The default RUDA forward reuses ruDNN chunk execution; backward recomputes same-device chunk operations and is not a standalone fused DeltaNet backward. Both output and final-state gradients propagate. Exact scale/normalization and empty-sequence behavior: [sequence implementation](../../ruda-torch/python/ruda_torch/sequence_training.py).

`learned_fake_quantize(x, scales, bits=4, block_shape=None, symmetric=True)` accepts nonempty FP32/FP16/BF16 input and same-device FP32 scales. Bits are 2, 4 or 8. No block shape means exactly one scale; otherwise each axis has a positive block size and scales number `product(ceil(shape/block_shape))`, including edge blocks. Scales are floored at `1e-8`; their gradients below the floor are zero. It returns a **floating** tensor with first-order straight-through derivatives, not packed inference weights. `LearnedFakeQuantize(initial_scales, ...)` stores a trainable, checkpointable FP32 scale parameter; keep it FP32 when moving the module. See [quantization.py](../../ruda-torch/python/ruda_torch/quantization.py) and [NF4 fine-tuning](finetuning.md) for the separate packed-base format.

## Compilation, graphs and models

- [General model compiler](model-compiler.md): `compile`, `make_backend`, `CompiledModel`, `CompiledFunction`, exact native overloads, shape/stream cache and error policies.
- [Fixed-address static graphs](static-pytorch-graphs.md): `StaticGraph`, `GraphOp`, replay/output lifetime and explicit first-order training.
- [Fine-tuning API](finetuning.md): all public packing, LoRA/NF4 loading, causal supervision, adapter and training-checkpoint entry points.
- [Architecture training guide](architecture-training.md): mHC, DSA/CSA/HCA, functional compressed KV caches, hybrid model construction and Python Muon groups.
- [Hybrid model source](../../ruda-torch/python/ruda_torch/hybrid_model.py): `MHCTransformerBlock`, `HybridAttentionLanguageModel`, `next_token_loss`; model composition rather than a pretrained-checkpoint loader.
- [mHC source](../../ruda-torch/python/ruda_torch/mhc.py): `MHC`, `MHCResidual`, `MHCSequential`, `MHCCoefficients`, `sinkhorn`.
- [Compressed attention source](../../ruda-torch/python/ruda_torch/sparse_attention.py): `LightningIndexer`/`DSAIndexer`, `indexer_kl_loss`, `LearnedKVCompressor`, `CompressedSparseAttention`/`CSA`, `HeavilyCompressedAttention`/`HCA`, `RotaryEmbedding`, `AttentionOutput`, `IndexerOutput`, `CompressedAttentionCache`, `CompressionState`.

Architecture components compose same-device tensor operations; their names do not imply compatibility with every pretrained DeepSeek architecture. API metadata errors normally raise `ValueError`/`TypeError`; native execution errors propagate, and asynchronous errors can surface during synchronization or readback.
