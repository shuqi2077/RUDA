# mHC, compressed attention and Python Muon

[Documentation](README.md) · [Native API](native-pytorch-api.md) · [中文](../zh/architecture-training.md)

These are trainable, same-device tensor compositions, not official DeepSeek pretrained-model loaders or FP4/FP8 attention kernels. Construct randomly initialized modules on CPU, then move them and inputs explicitly to the execution device. FP16/BF16 work uses FP32 mapping/score arithmetic; explicit FP64 inputs retain FP64 where supported. Device operator availability still determines whether a composed model can execute.

## Hyper-connections and residual-free branches

`MHC(width, streams=4, sinkhorn_iterations=20, eps=1e-6, gate_init=0.01, device=None, dtype=None)` owns trainable mapping, gate and bias parameters. Width/streams/iteration count are positive integers, epsilon positive and gate initialization finite. State layout is `[...,streams,width]`; parameters and state share a device.

| Method / type | Contract |
| --- | --- |
| `expand(x)` | `[...,width]` → cloned `[...,streams,width]` stream state. |
| `reduce(state)` | Average the stream axis → `[...,width]`. |
| `coefficients(state)` | `MHCCoefficients(pre, post, residual)` with shapes `[...,streams]`, `[...,streams]`, `[...,streams,streams]`. |
| `pre(state)` | Return merged `[...,width]` branch input and its coefficients. |
| `post(state, branch_output, coefficients)` | Combine residual mapping and branch update; branch output preserves all dimensions except streams. |
| `forward(state, branch, *args, **kwargs)` | Apply pre, a residual-free branch, then post. |
| `sinkhorn(logits, iterations=20)` | Nonempty square trailing dimensions; log-space column then row normalization. Finite iterations approximate column normalization, not an exact manifold projection. |

`MHCResidual(branch, width, streams=4, checkpoint_branch=False, **mhc_options)` wraps an `nn.Module` that **does not add its own residual input**. `checkpoint_branch=True` uses non-reentrant branch recomputation while training with gradients. `MHCSequential(width, branches, streams=4, **options)` requires at least one such branch, expands once and averages streams after all layers. Parameters participate normally in `state_dict`, autograd and optimizers.

```python
import torch
from torch import nn
import ruda_torch as r

block = r.MHCSequential(64, [nn.Sequential(nn.Linear(64, 64), nn.SiLU())], streams=4)
block = block.to('ruda:0')
x = torch.ones(2, 8, 64).to('ruda:0').requires_grad_()
y = block(x)  # [2,8,64]; the branch must not add x again
```

## DSA lightning indexer

`LightningIndexer` and `DSAIndexer` are aliases. Required width is positive; defaults are `num_heads=4, head_dim=16, query_dim=None, topk=32, query_chunk_size=32, key_chunk_size=128, external_keys=False, detach_inputs=True, rope_dim=0, eps=1e-6`. Omitted query dimension uses width. RoPE dimension is even, nonnegative and no greater than head dimension.

Inputs are `[B,T,width]`; optional query latent `[B,T,query_dim]`; prepared keys `[B,S,head_dim]`. `project_keys(x, pos=None)` creates token keys unless `external_keys=True`. `scores` returns dense `[B,T,S]` scores for diagnostics/warm-up; selection instead streams query/key chunks. Chunking bounds intermediate storage, not the quadratic scoring arithmetic.

`select` returns discrete top-k int64 indices; invalid slots are `-1`. Its masks are same-device Bool: `allowed [B,T,S]`, `key_valid [B,S]`, `query_valid [B,T]`; absolute `query_positions [T]` and `key_end_positions [S]` are int64. An entry is causally visible when its end position does not exceed the query position. `forward(..., causal=True)` returns `IndexerOutput(indices, scores, valid)`, with `valid = indices >= 0`; external compressed keys need their actual absolute end positions. `causal=False` explicitly selects noncausal/cross-attention indexing.

Top-k decisions have no gradient. Input features are detached by default, but indexer parameters train through recomputed selected scores and `indexer_kl_loss(scores, teacher, valid=None)`. Teacher mass is `[B,T,S]` or `[B,T,H,S]`, nonnegative and detached; head masses are summed and empty rows contribute zero. `distillation_loss(..., indices=None)` uses dense scores, or recomputes selected scores when indices are supplied. Add the auxiliary loss explicitly; language-model loss alone does not train the discrete selector.

## CSA, HCA and compression parameters

`CompressedSparseAttention`/`CSA(width, num_heads, **options)` combines overlapping learned block compression, DSA selection and a local window. `HeavilyCompressedAttention`/`HCA` uses nonoverlapping compression and all causally visible compressed entries, without an indexer. Both use one softmax over local and compressed entries; only completed compressed blocks are visible.

| Option | Default / restriction |
| --- | --- |
| `head_dim` | `width/num_heads`; heads must divide width when omitted. |
| `compress_ratio` | CSA 4, HCA 128; positive integer. |
| `topk`, `window_size` | 32 selected compressed entries (CSA), local window 128. |
| `query_rank` | width; positive query bottleneck size. |
| `index_heads`, `index_dim` | 4 and 16 for CSA's indexer. |
| `rope_dim`, `rope_base` | 0 and 10000; dimension even and fitting attention/indexer channels. Base RoPE, not YaRN. |
| `output_groups`, `output_rank` | 1 and `width/output_groups`; groups divide head count, rank positive. |
| `query_chunk_size`, `key_chunk_size` | 32 and 128; positive. Key chunk size controls CSA indexing. |
| `attention_sink`, `eps` | True and `1e-6`; sink is a learned denominator contribution, not another KV value. |

`forward(x, valid_mask=None, return_aux=False, indexer_warmup=False)` accepts nonempty `[B,T,width]`, with same-device Bool `[B,T]` mask, True for valid positions. Output has the same feature shape. `return_aux=True` returns `AttentionOutput(output, indexer_loss)`. `indexer_warmup=True` attends all visible compressed entries and trains the indexer against detached attention mass; it retains dense compressed-attention cost. For indexer-only warm-up, backpropagate auxiliary loss alone. HCA returns a disconnected zero auxiliary loss.

`LearnedKVCompressor(width, head_dim, ratio, overlap=False, eps=1e-6, ...)` returns `(compressed, valid)` for full-sequence `[B,T,width]`, with `floor(T/ratio)` entries. Incomplete tails are not emitted; entirely invalid blocks are zero. Overlap uses distinct learned previous/current block paths. `RotaryEmbedding(rope_dim, base=10000., device=None)` acts on trailing channels of `[B,T,D]` or `[B,T,H,D]` with `[T]` positions; its nonpersistent frequencies remain FP32 during dtype moves. `inverse=True` reverses rotation.

## Prefill, decode and cache ownership

Cached operations require **both** `eval()` and `no_grad()`/`inference_mode()`. Training uses the differentiable full-sequence forward. `attention.forward_cached(x, cache=None, valid_mask=None)` accepts any nonempty chunk size and returns `(output, new_cache)`. Pass the returned cache to the next chunk; do not replace it with another layer's cache.

```python
attention = r.CSA(64, 4, compress_ratio=4, window_size=16).to('ruda:0').eval()
with torch.no_grad():
    prefill, cache = attention.forward_cached(x[:, :6].detach())
    decode, cache = attention.forward_cached(x[:, 6:].detach(), cache)
    retained_bytes = cache.tensor_bytes
```

`CompressedAttentionCache` records owner identity, parameter versions, seen positions, compressed history/validity, optional index keys, the last `window_size-1` local entries and compressor tails. Changing parameters, layer, batch size, dtype or device invalidates reuse and raises an error; start with `cache=None` again. Cache evolution is functional: prior cache objects are not mutated.

`cache.reorder(batch_indices)` returns reordered/forked beam state using a same-device one-dimensional int64 index tensor. `tensor_bytes` counts logically retained tensors, not allocator peak or a fixed capacity; compressed history still grows with sequence length. `LearnedKVCompressor.append(x, valid_mask=None, state=None)` similarly returns compressed additions, validity and a new `CompressionState`; its tails are cloned so a small tail does not retain a whole prompt allocation.

## Mixed model and losses

`MHCTransformerBlock(width, num_heads, streams=4, attention_kind='csa', feedforward_width=None, sinkhorn_iterations=20, eps=1e-6, **attention_options)` maps `[B,T,streams,width]` to the same shape using attention and a gated FFN with independent mHC connections. Default FFN width is `4*width`; attention kind is `csa` or `hca`.

`HybridAttentionLanguageModel(vocab_size, width, num_heads, num_layers, streams=4, csa_ratio=4, hca_ratio=128, tie_embeddings=False, ...)` alternates CSA/HCA and returns `[B,T,V]` logits for nonempty `[B,T]` int32/int64 tokens. Its optional Bool valid mask is `[B,T]`. `return_aux=True` adds the sum of indexer losses. These sizes are caller choices, not published large-model hyperparameters. The whole model does not expose a combined `forward_cached`; individual attention layers do.

`next_token_loss(logits, tokens, valid_mask=None)` averages shifted cross entropy over source/target pairs where **both** mask entries are valid. Length-one or wholly invalid batches yield connected zero. Even masked token positions must contain valid vocabulary IDs, not `-100`; use [chunked fine-tuning loss](finetuning.md) for explicit ignored labels. These are different supervision contracts.

## Python Muon and explicit AdamW exclusions

`Muon(params, lr=0.02, momentum=0.95, weight_decay=0., nesterov=True, momentum_mode='sgd', dampening=0., ns_steps=5, ns_coefficients=(3.4445,-4.775,2.0315), eps=1e-7, adjust_lr='original', matrix_layout='as_stored', stable_normalization=True, flatten=False, max_grad_norm=None)` is single-device eager optimization. Dense same-shape/dtype/device gradients are required; Muon matrices are 2-D unless explicit `flatten=True` reshapes higher-rank parameters to `[first_dimension,-1]`.

- `ns_steps` is 1–99; coefficients are three finite values. `muon_orthogonalize` performs finite quintic Newton–Schulz iterations, not exact polar decomposition. FP16/BF16 masters and moments are FP32.
- `momentum_mode` is `sgd` or `ema`; Nesterov requires positive momentum and zero dampening, and EMA requires zero dampening.
- `adjust_lr='original'` uses `sqrt(max(1,rows/cols))`; `match_rms_adamw` uses `0.2*sqrt(max(rows,cols))`. `matrix_layout='input_output'` swaps those row/column semantics for the scale; it does not transpose stored parameters.
- Optional positive `max_grad_norm` clips unscaled gradients internally; stored gradients remain unchanged. Finite-status readbacks make the step noncapturable. Nonfinite active gradients/proposals skip the whole update without decay/counter advancement; device failure during commit is not transactional.
- State tracks master weights, moments, algorithm and parameter versions. External parameter mutation after state creation requires resetting/reloading matching state. Synchronize distributed gradients before orthogonalizing **complete matrices**, not local TP/FSDP shards.

`MuonAdamW.from_model` selects module objects, not guessed names:

```python
model = r.HybridAttentionLanguageModel(256, 64, 4, 2, hca_ratio=8).to('ruda:0')
optimizer = r.MuonAdamW.from_model(
    model, muon_modules=list(model.layers),
    adamw_modules=[model.embedding, model.head], lr=0.02, adamw_lr=0.001,
)
```

Only nominated trainable matrices enter Muon; embeddings, explicit exclusions and all other trainable parameters use AdamW. Shared parameters occur once, with AdamW exclusion taking precedence. No eligible nominated matrix is an error. The manual constructor requires explicit `use_muon=True/False` group entries. Save model, optimizer and any scaler together; do not substitute the Rust Muon record for this Python state format.

Sources: [mHC](../../ruda-torch/python/ruda_torch/mhc.py), [indexing/compression/cache](../../ruda-torch/python/ruda_torch/sparse_attention.py), [hybrid model](../../ruda-torch/python/ruda_torch/hybrid_model.py), [Python Muon](../../ruda-torch/python/ruda_torch/optim.py). The existing [composition example](../../ruda-torch/python/examples/train_hybrid_attention.py) is separate from training a real pretrained checkpoint or establishing a performance baseline.
