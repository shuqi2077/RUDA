"""Trainable CSA/HCA and DSA lightning-indexer components.

Algorithmic references: arXiv:2606.19348, arXiv:2512.02556 and DeepSeek's
published inference implementation. This module implements floating-point,
composed operators, not the official FP4/FP8 kernels or checkpoint layout.

CSA pools the current and previous block with distinct learned projections;
HCA pools one non-overlapping block. Both combine compressed KV with an exact
local window in ONE softmax. Only completed blocks are visible. Cache mutation
is functional and eval-only; training uses the differentiable full sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F

from ._architecture_ops import (positive_int, positive_float, positions, work_dtype,
    precision_context, rms, masked_softmax, stable_topk, gather_entries)


class AttentionOutput(NamedTuple):
    output: torch.Tensor
    indexer_loss: torch.Tensor


class IndexerOutput(NamedTuple):
    indices: torch.Tensor
    scores: torch.Tensor
    valid: torch.Tensor


def indexer_kl_loss(scores: torch.Tensor, teacher: torch.Tensor,
                    valid: torch.Tensor | None = None) -> torch.Tensor:
    """Mean KL(teacher || indexer), ignoring empty rows; teacher is detached.

    teacher is nonnegative attention mass [B,T,S], or [B,T,H,S] (heads are
    summed). Zero-mass rows contribute zero. Inputs/teacher features may be
    detached separately by LightningIndexer; this loss never trains the teacher.
    """
    if teacher.ndim == scores.ndim + 1:
        teacher = teacher.sum(dim=-2)
    if teacher.shape != scores.shape or teacher.device != scores.device:
        raise ValueError("teacher must match scores (optionally with an attention-head axis)")
    dtype = work_dtype(scores)
    if valid is None:
        valid = torch.ones_like(scores, dtype=torch.bool)
    if valid.shape != scores.shape or valid.dtype != torch.bool or valid.device != scores.device:
        raise ValueError("valid must be a same-shape boolean mask on the scores' device")
    if scores.shape[-1] == 0:
        return scores.sum() * 0
    with precision_context(scores):
        target = teacher.detach().to(dtype) * valid.to(dtype)
        mass = target.sum(-1, keepdim=True)
        active = (mass > 0) & valid.any(-1, keepdim=True)
        target = target / torch.where(mass > 0, mass, torch.ones_like(mass))
        logits = scores.to(dtype).masked_fill(~valid, float("-inf"))
        logits = torch.where(active, logits, torch.zeros_like(logits))
        logp = F.log_softmax(logits, dim=-1)
        logp = torch.where(valid & active, logp, torch.zeros_like(logp))
        logt = torch.where(target > 0, target, torch.ones_like(target)).log()
        rows = (target * (logt - logp)).sum(-1)
        count = active.to(dtype).sum()
        return rows.sum() / torch.where(count > 0, count, torch.ones_like(count))


class RotaryEmbedding(nn.Module):
    """Interleaved real-valued rotary embedding on the trailing rope_dim channels.

    Base RoPE only, without YaRN context scaling. The same rotated KV is used
    as key and value; the attention output is de-rotated at the query position.
    """
    def __init__(self, rope_dim: int, base: float = 10000.0, *, device=None):
        super().__init__()
        if isinstance(rope_dim, bool) or not isinstance(rope_dim, int) or rope_dim < 0 or rope_dim % 2:
            raise ValueError("rope_dim must be a nonnegative even integer")
        self.rope_dim = rope_dim
        positive_float(base, "rope_base")
        self._frequency_values = tuple(base ** (-i / rope_dim) for i in range(0, rope_dim, 2))
        self.register_buffer("frequencies", torch.tensor(self._frequency_values, dtype=torch.float32, device=device), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Module.half()/bfloat16() must not permanently round positional
        # frequencies. Reconstruct during the explicit device/dtype move, not
        # inside attention execution, and keep the buffer on its moved device.
        if self.frequencies.dtype != torch.float32:
            self.frequencies = torch.tensor(self._frequency_values, device=self.frequencies.device, dtype=torch.float32)
        return self

    def forward(self, x: torch.Tensor, pos: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        d = self.rope_dim
        if d == 0:
            return x
        if x.shape[-1] < d or x.ndim not in (3, 4) or pos.shape != (x.shape[1],):
            raise ValueError("rotary inputs must be [B,T,D] or [B,T,H,D] with T positions")
        if pos.device != x.device or self.frequencies.device != x.device:
            raise ValueError("move rotary buffers and positions to the input device before execution")
        with precision_context(x):
            dtype = work_dtype(x)
            angles = pos.to(dtype).unsqueeze(-1) * self.frequencies.to(dtype=dtype)
            if inverse:
                angles = -angles
            shape = (1, x.shape[1], *((1,) if x.ndim == 4 else ()), d // 2)
            cosine, sine = angles.cos().reshape(shape), angles.sin().reshape(shape)
            rotated = x[..., -d:].to(dtype)
            even, odd = rotated[..., ::2], rotated[..., 1::2]
            rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)
        return torch.cat((x[..., :-d], rotated.to(x.dtype)), -1)


class LightningIndexer(nn.Module):
    """DSA score = sum_h(weight_h * relu(q_h dot k)), plus trainable KL loss.

    Input features are detached by default, matching separate indexer training.
    Selection is discrete: the language-model loss alone does not train it.
    Add the returned/aligned KL loss, or call distillation_loss during warm-up.
    Keys can be token projections (DSA) or externally prepared compressed keys
    (CSA). Selection streams query/key chunks instead of retaining a dense
    sequence-by-sequence score tensor. Arithmetic remains quadratic in length.
    """
    def __init__(self, width: int, num_heads: int = 4, head_dim: int = 16, *,
                 query_dim: int | None = None, topk: int = 32,
                 query_chunk_size: int = 32, key_chunk_size: int = 128,
                 external_keys: bool = False, detach_inputs: bool = True,
                 rope_dim: int = 0, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.width = positive_int(width, "width")
        self.num_heads = positive_int(num_heads, "num_heads")
        self.head_dim = positive_int(head_dim, "head_dim")
        self.query_dim = positive_int(query_dim if query_dim is not None else width, "query_dim")
        self.topk = positive_int(topk, "topk")
        self.query_chunk_size = positive_int(query_chunk_size, "query_chunk_size")
        self.key_chunk_size = positive_int(key_chunk_size, "key_chunk_size")
        self.detach_inputs, self.external_keys = bool(detach_inputs), bool(external_keys)
        self.eps = positive_float(eps, "eps")
        opts = dict(device=device, dtype=dtype)
        self.query = nn.Linear(self.query_dim, num_heads * head_dim, bias=False, **opts)
        self.head_weight = nn.Linear(width, num_heads, bias=False, **opts)
        self.key = None if external_keys else nn.Linear(width, head_dim, bias=False, **opts)
        self.key_norm = None if external_keys else nn.LayerNorm(head_dim, eps=eps, **opts)
        self.rotary = RotaryEmbedding(rope_dim, device=device)
        if rope_dim > head_dim:
            raise ValueError("indexer rope_dim exceeds head_dim")
        self.score_scale = (head_dim * num_heads) ** -0.5

    def project_keys(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        if self.key is None:
            raise ValueError("external_keys=True: provide prepared indexer keys")
        if self.detach_inputs:
            x = x.detach()
        result = self.key_norm(self.key(x))
        if pos is None:
            pos = positions(x.shape[1], x.device)
        return self.rotary(result, pos)

    def _queries(self, x, query_latent=None, pos=None):
        latent = x if query_latent is None else query_latent
        if x.ndim != 3 or x.shape[-1] != self.width or latent.shape != (*x.shape[:2], self.query_dim):
            raise ValueError("indexer inputs must be [B,T,width] with matching query_latent")
        if x.device != self.query.weight.device or latent.device != x.device:
            raise ValueError("indexer features and parameters must be on the same device")
        if self.detach_inputs:
            x, latent = x.detach(), latent.detach()
        dtype = work_dtype(x)
        with precision_context(x):
            q = F.linear(latent.to(dtype), self.query.weight.to(dtype)).reshape(
                *x.shape[:2], self.num_heads, self.head_dim)
            weight = F.linear(x.to(dtype), self.head_weight.weight.to(dtype)) * self.score_scale
            if pos is None:
                pos = positions(x.shape[1], x.device)
            q = self.rotary(q, pos)
        return q, weight

    def _score(self, q, weight, keys):
        if keys.ndim != 3 or keys.shape[0] != q.shape[0] or keys.shape[-1] != self.head_dim:
            raise ValueError("prepared keys must have shape [batch, entries, head_dim]")
        if keys.device != q.device:
            raise ValueError("prepared keys must be on the query device")
        dots = torch.matmul(q.transpose(1, 2), keys.to(q.dtype).transpose(1, 2).unsqueeze(1))
        return (dots.relu() * weight.transpose(1, 2).unsqueeze(-1)).sum(1)

    def scores(self, x, keys, *, query_latent=None, query_positions=None):
        """Dense score API for explicit warm-up/diagnostics, not used by selection."""
        q, w = self._queries(x, query_latent, query_positions)
        with precision_context(x):
            return self._score(q, w, keys)

    def selected_scores(self, x, keys, indices, *, query_latent=None, query_positions=None):
        q, weight = self._queries(x, query_latent, query_positions)
        with precision_context(x):
            selected = gather_entries(keys.to(q.dtype), indices)
            dots = q @ selected.transpose(-1, -2)
            return (dots.relu() * weight.unsqueeze(-1)).sum(-2)

    def select(self, x, keys, *, allowed=None, query_latent=None, query_positions=None,
               key_valid=None, query_valid=None, key_end_positions=None):
        """Stream top-k; causal metadata avoids allocating a full [B,T,S] mask."""
        if x.ndim != 3 or keys.ndim != 3 or keys.shape[0] != x.shape[0] or keys.shape[-1] != self.head_dim:
            raise ValueError("indexer expects [B,T,width] features and [B,S,head_dim] keys")
        if keys.device != x.device:
            raise ValueError("indexer keys must be on the feature device")
        b, t, _ = x.shape
        s = keys.shape[1]
        for name, value, shape, dtype in (
            ('key_valid', key_valid, (b, s), torch.bool),
            ('query_valid', query_valid, (b, t), torch.bool),
            ('key_end_positions', key_end_positions, (s,), torch.int64),
            ('query_positions', query_positions, (t,), torch.int64)):
            if value is not None and (value.shape != shape or value.dtype != dtype or value.device != x.device):
                raise ValueError(f'{name} must have shape {shape}, dtype {dtype}, on the feature device')
        if allowed is not None and (allowed.shape != (b, t, s) or allowed.dtype != torch.bool or allowed.device != x.device):
            raise ValueError("allowed must be a same-device [B,T,S] boolean mask")
        if query_positions is None:
            query_positions = positions(t, x.device)
        # No selection graph is retained. Gradients are computed from scores of
        # the chosen entries, not through top-k's integer decisions.
        chunks = []
        with torch.no_grad(), precision_context(x):
            q, weights = self._queries(x, query_latent, query_positions)
            for begin in range(0, t, self.query_chunk_size):
                end = min(t, begin + self.query_chunk_size)
                best = q.new_empty((b, end - begin, 0))
                ids = torch.empty((b, end - begin, 0), dtype=torch.int64, device=x.device)
                for start in range(0, s, self.key_chunk_size):
                    stop = min(s, start + self.key_chunk_size)
                    score = self._score(q[:, begin:end], weights[:, begin:end], keys[:, start:stop])
                    valid = torch.ones_like(score, dtype=torch.bool)
                    if allowed is not None:
                        valid = valid & allowed[:, begin:end, start:stop]
                    if key_valid is not None:
                        valid = valid & key_valid[:, None, start:stop]
                    if query_valid is not None:
                        valid = valid & query_valid[:, begin:end, None]
                    if key_end_positions is not None:
                        valid = valid & (key_end_positions[None, None, start:stop] <= query_positions[None, begin:end, None])
                    new_ids = positions(stop - start, x.device, start).reshape(1, 1, -1).expand_as(score)
                    best, ids = stable_topk(torch.cat((best, score), -1), self.topk,
                        indices=torch.cat((ids, new_ids), -1),
                        valid=torch.cat((ids >= 0, valid), -1))
                chunks.append(ids)
        if not chunks:
            return torch.empty((b, 0, min(self.topk, s)), device=x.device, dtype=torch.int64)
        return torch.cat(chunks, 1)

    def forward(self, x, key_states=None, *, prepared_keys=None, allowed=None,
                query_latent=None, query_positions=None, causal=True,
                key_end_positions=None, key_valid=None, query_valid=None):
        """Causal token indexing by default; set causal=False for cross-attention.

        External compressed/cache keys require their absolute end positions.
        Without metadata, causal keys are interpreted as tokens numbered from 0.
        """
        if type(causal) is not bool:
            raise TypeError('causal must be bool')
        if prepared_keys is None:
            prepared_keys = self.project_keys(x if key_states is None else key_states)
        if causal and key_end_positions is None:
            key_end_positions = positions(prepared_keys.shape[1], x.device)
        indices = self.select(x, prepared_keys, allowed=allowed, query_latent=query_latent,
                              query_positions=query_positions, key_end_positions=key_end_positions,
                              key_valid=key_valid, query_valid=query_valid)
        scores = self.selected_scores(x, prepared_keys, indices, query_latent=query_latent,
                                      query_positions=query_positions)
        return IndexerOutput(indices, scores, indices >= 0)

    def distillation_loss(self, x, prepared_keys, teacher, *, allowed=None,
                          indices=None, query_latent=None, query_positions=None):
        if indices is None:
            scores = self.scores(x, prepared_keys, query_latent=query_latent, query_positions=query_positions)
            return indexer_kl_loss(scores, teacher, allowed)
        scores = self.selected_scores(x, prepared_keys, indices, query_latent=query_latent,
                                      query_positions=query_positions)
        return indexer_kl_loss(scores, teacher, indices >= 0)


DSAIndexer = LightningIndexer


@dataclass(frozen=True)
class CompressionState:
    tail: torch.Tensor
    tail_valid: torch.Tensor
    previous: torch.Tensor
    previous_valid: torch.Tensor

    def reorder(self, batch_indices: torch.Tensor):
        return CompressionState(*(x.index_select(0, batch_indices) for x in
            (self.tail, self.tail_valid, self.previous, self.previous_valid)))


class LearnedKVCompressor(nn.Module):
    """Per-channel gated pooling, with optional previous-block overlap.

    The incomplete tail is NOT emitted. Empty/padded blocks produce exact zero.
    Positional biases distinguish offsets inside each block. FP32 arithmetic is
    used for low-precision activations and FP64 is retained for gradcheck.
    """
    def __init__(self, width: int, head_dim: int, ratio: int, *, overlap=False,
                 eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.width = positive_int(width, "width")
        self.head_dim = positive_int(head_dim, "head_dim")
        self.ratio = positive_int(ratio, "ratio")
        self.overlap = bool(overlap)
        self.eps = positive_float(eps, "eps")
        opts = dict(device=device, dtype=dtype or torch.float32)
        streams = 2 if overlap else 1
        self.value = nn.Linear(width, streams * head_dim, bias=False, **opts)
        self.gate = nn.Linear(width, streams * head_dim, bias=False, **opts)
        self.position_bias = nn.Parameter(torch.zeros(ratio, streams * head_dim, **opts))
        self.norm_weight = nn.Parameter(torch.ones(head_dim, **opts))

    def _compress(self, x, valid, previous=None, previous_valid=None):
        b, length, _ = x.shape
        r, d = self.ratio, self.head_dim
        blocks = length // r
        dtype = work_dtype(x)
        if blocks == 0:
            return x.new_zeros((b, 0, d)) + x.sum() * 0, valid[:, :0]
        with precision_context(x):
            complete = x[:, :blocks*r].to(dtype)
            values = F.linear(complete, self.value.weight.to(dtype)).reshape(b, blocks, r, -1)
            gates = F.linear(complete, self.gate.weight.to(dtype)).reshape(b, blocks, r, -1)
            gates = gates + self.position_bias.to(dtype)
            mask = valid[:, :blocks*r].reshape(b, blocks, r, 1)
            if self.overlap:
                # First d channels are the previous-block path, as in the
                # official compressor; remaining d are the current-block path.
                if previous is None:
                    pv = values.new_zeros((b, 1, r, d))
                    pg = gates.new_zeros((b, 1, r, d))
                    pm = torch.zeros((b, 1, r, 1), dtype=torch.bool, device=x.device)
                else:
                    pv = F.linear(previous.to(dtype), self.value.weight[:d].to(dtype)).unsqueeze(1)
                    pg = (F.linear(previous.to(dtype), self.gate.weight[:d].to(dtype)) + self.position_bias[:, :d].to(dtype)).unsqueeze(1)
                    pm = previous_valid.reshape(b, 1, r, 1)
                prev_v = torch.cat((pv, values[:, :-1, :, :d]), 1)
                prev_g = torch.cat((pg, gates[:, :-1, :, :d]), 1)
                prev_m = torch.cat((pm, mask[:, :-1]), 1)
                values = torch.cat((prev_v, values[..., d:]), 2)
                gates = torch.cat((prev_g, gates[..., d:]), 2)
                mask = torch.cat((prev_m, mask), 2)
            weights = masked_softmax(gates, mask, dim=2)
            pooled = (values * weights).sum(2)
            pooled = rms(pooled, self.eps, self.norm_weight.to(dtype))
        return pooled.to(x.dtype), mask.any(2).squeeze(-1)

    def forward(self, x, valid_mask=None):
        if x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError("compressor input must be [batch, length, width]")
        if x.device != self.value.weight.device:
            raise ValueError("compressor input and parameters must share a device")
        valid = _valid_mask(x, valid_mask)
        return self._compress(x, valid)

    def append(self, x, valid_mask=None, state: CompressionState | None = None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("compression cache requires eval() and no_grad()/inference_mode()")
        work_dtype(x)
        if x.ndim != 3 or x.shape[-1] != self.width or x.device != self.value.weight.device:
            raise ValueError('compressor input must be [batch, length, width] on the parameter device')
        valid = _valid_mask(x, valid_mask)
        if state is None:
            data, masks, prev, prev_valid = x, valid, None, None
        else:
            if state.tail.shape[0] != x.shape[0] or state.tail.device != x.device or state.tail.dtype != x.dtype:
                raise ValueError("compression cache batch/device/dtype mismatch")
            data = torch.cat((state.tail, x), 1)
            masks = torch.cat((state.tail_valid, valid), 1)
            prev, prev_valid = state.previous, state.previous_valid
        result, result_valid = self._compress(data, masks, prev, prev_valid)
        cutoff = data.shape[1] // self.ratio * self.ratio
        if cutoff:
            prev = data[:, cutoff-self.ratio:cutoff]
            prev_valid = masks[:, cutoff-self.ratio:cutoff]
        elif prev is None:
            prev = x.new_zeros((x.shape[0], self.ratio, self.width))
            prev_valid = torch.zeros((x.shape[0], self.ratio), device=x.device, dtype=torch.bool)
        # Clone all retained slices: otherwise a tiny tail pins the whole prompt.
        new = CompressionState(data[:, cutoff:].detach().clone(), masks[:, cutoff:].detach().clone(),
                               prev.detach().clone(), prev_valid.detach().clone())
        return result, result_valid, new


def _valid_mask(x, mask):
    if mask is None:
        return torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
    if mask.shape != x.shape[:2] or mask.dtype != torch.bool or mask.device != x.device:
        raise ValueError("valid_mask must be [batch, length], bool, on the input device (True = valid)")
    return mask


@dataclass(frozen=True)
class CompressedAttentionCache:
    owner: int
    parameter_versions: tuple
    seen: int
    compressed: torch.Tensor
    compressed_valid: torch.Tensor
    index_keys: torch.Tensor | None
    local: torch.Tensor
    local_valid: torch.Tensor
    compression: CompressionState
    index_compression: CompressionState | None

    def reorder(self, batch_indices: torch.Tensor):
        """Return a reordered/forked beam cache; no source cache is modified."""
        if batch_indices.ndim != 1 or batch_indices.dtype != torch.int64 or batch_indices.device != self.local.device:
            raise ValueError("beam indices must be same-device int64 vector")
        select = lambda x: None if x is None else x.index_select(0, batch_indices)
        return CompressedAttentionCache(self.owner, self.parameter_versions, self.seen,
            select(self.compressed), select(self.compressed_valid), select(self.index_keys),
            select(self.local), select(self.local_valid), self.compression.reorder(batch_indices),
            None if self.index_compression is None else self.index_compression.reorder(batch_indices))

    @property
    def tensor_bytes(self):
        """Logical retained tensor bytes, not allocator peak memory."""
        tensors = [self.compressed, self.compressed_valid, self.index_keys, self.local, self.local_valid]
        for state in (self.compression, self.index_compression):
            if state is not None:
                tensors.extend((state.tail, state.tail_valid, state.previous, state.previous_valid))
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)


class _CompressedAttention(nn.Module):
    def __init__(self, width, num_heads, *, sparse, head_dim=None, compress_ratio=4,
                 topk=32, window_size=128, query_rank=None, index_heads=4, index_dim=16,
                 rope_dim=0, rope_base=10000.0, output_groups=1, output_rank=None,
                 query_chunk_size=32, key_chunk_size=128, attention_sink=True,
                 eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.width = positive_int(width, "width")
        self.num_heads = positive_int(num_heads, "num_heads")
        if head_dim is None and width % num_heads:
            raise ValueError("num_heads must divide width when head_dim is omitted")
        self.head_dim = positive_int(head_dim if head_dim is not None else width // num_heads, "head_dim")
        self.ratio = positive_int(compress_ratio, "compress_ratio")
        self.window_size = positive_int(window_size, "window_size")
        self.query_chunk_size = positive_int(query_chunk_size, "query_chunk_size")
        self.query_rank = positive_int(query_rank if query_rank is not None else width, "query_rank")
        self.output_groups = positive_int(output_groups, "output_groups")
        if num_heads % self.output_groups:
            raise ValueError("output_groups must divide num_heads")
        self.output_rank = positive_int(output_rank if output_rank is not None else width // self.output_groups, "output_rank")
        self.sparse = bool(sparse)
        self.eps = positive_float(eps, "eps")
        if rope_dim > self.head_dim or (sparse and rope_dim > index_dim):
            raise ValueError("rope_dim must fit attention and indexer head dimensions")
        opts = dict(device=device, dtype=dtype)
        self.query_down = nn.Linear(width, self.query_rank, bias=False, **opts)
        self.query_up = nn.Linear(self.query_rank, num_heads*self.head_dim, bias=False, **opts)
        self.query_norm = nn.Parameter(torch.ones(self.query_rank, **opts))
        self.local_kv = nn.Linear(width, self.head_dim, bias=False, **opts)
        self.local_norm = nn.Parameter(torch.ones(self.head_dim, **opts))
        self.compressor = LearnedKVCompressor(width, self.head_dim, self.ratio, overlap=sparse, eps=eps, **opts)
        self.rotary = RotaryEmbedding(rope_dim, rope_base, device=device)
        self.output_down = nn.ModuleList([nn.Linear(num_heads//self.output_groups*self.head_dim,
            self.output_rank, bias=False, **opts) for _ in range(self.output_groups)])
        self.output_up = nn.Linear(self.output_groups*self.output_rank, width, bias=False, **opts)
        if attention_sink:
            self.sink = nn.Parameter(torch.zeros(num_heads, device=device, dtype=torch.float32))
        else:
            self.register_parameter("sink", None)
        if sparse:
            self.indexer = LightningIndexer(width, index_heads, index_dim, query_dim=self.query_rank,
                topk=topk, query_chunk_size=query_chunk_size, key_chunk_size=key_chunk_size,
                external_keys=True, rope_dim=rope_dim, eps=eps, **opts)
            self.index_compressor = LearnedKVCompressor(width, index_dim, self.ratio, overlap=True, eps=eps, **opts)
        else:
            self.indexer = None
            self.index_compressor = None

    def _validate(self, x, valid_mask):
        work_dtype(x)
        if x.ndim != 3 or x.shape[-1] != self.width or min(x.shape[:2]) < 1:
            raise ValueError("attention requires nonempty [batch, length, width] input")
        if x.device != self.query_down.weight.device:
            raise ValueError("attention features and parameters must share a device")
        return _valid_mask(x, valid_mask)

    def _project(self, x, pos):
        latent = rms(self.query_down(x), self.eps, self.query_norm)
        q = self.query_up(latent).reshape(*x.shape[:2], self.num_heads, self.head_dim)
        q = self.rotary(rms(q, self.eps).to(x.dtype), pos)
        local = self.rotary(rms(self.local_kv(x), self.eps, self.local_norm).to(x.dtype), pos)
        return latent, q, local

    def _attend(self, x, latent, q, query_pos, query_valid, local, local_valid,
                local_start, compressed, compressed_valid, index_keys, *, return_aux, indexer_warmup=False):
        b, t, _ = x.shape
        s = compressed.shape[1]
        ends = positions(s, x.device) * self.ratio + self.ratio - 1
        output_chunks, loss_chunks, active_chunks = [], [], []
        for begin in range(0, t, self.query_chunk_size):
            end = min(t, begin + self.query_chunk_size)
            pos = query_pos[begin:end]
            qvalid = query_valid[:, begin:end]
            # Fixed-width local gather. Invalid slots use -1, never a real key.
            local_ids = pos[:, None] - (self.window_size - 1) + positions(self.window_size, x.device)[None, :]
            local_ids = local_ids - local_start
            lvalid = (local_ids >= 0) & (local_ids < local.shape[1])
            local_ids = torch.where(lvalid, local_ids, torch.full_like(local_ids, -1))
            local_ids = local_ids[None].expand(b, -1, -1)
            local_entries = gather_entries(local, local_ids)
            lmask = gather_entries(local_valid.unsqueeze(-1).to(x.dtype), local_ids).squeeze(-1) > 0
            lmask = lmask & qvalid.unsqueeze(-1)
            if self.indexer is not None and not indexer_warmup:
                ids = self.indexer.select(x[:, begin:end], index_keys,
                    query_latent=latent[:, begin:end], query_positions=pos,
                    key_valid=compressed_valid, query_valid=qvalid, key_end_positions=ends)
                cmask = ids >= 0
            else:
                ids = positions(s, x.device).reshape(1, 1, s).expand(b, end-begin, s)
                cmask = compressed_valid[:, None, :] & qvalid.unsqueeze(-1) & (ends[None, None, :] <= pos[None, :, None])
                ids = torch.where(cmask, ids, torch.full_like(ids, -1))
            selected = gather_entries(compressed, ids)
            entries = torch.cat((local_entries, selected), dim=2)
            mask = torch.cat((lmask, cmask), dim=-1)
            with precision_context(x):
                dtype = work_dtype(x)
                scores = (q[:, begin:end].to(dtype) @ entries.to(dtype).transpose(-1, -2)) * self.head_dim**-0.5
                allowed = mask.unsqueeze(-2).expand_as(scores)
                if self.sink is not None:
                    sink = self.sink.to(dtype).reshape(1, 1, self.num_heads, 1).expand(b, end-begin, -1, -1)
                    scores = torch.cat((scores, sink), -1)
                    allowed = torch.cat((allowed, torch.ones_like(sink, dtype=torch.bool)), -1)
                probabilities = masked_softmax(scores, allowed)
                if self.sink is not None:
                    probabilities = probabilities[..., :-1]
                attended = probabilities @ entries.to(dtype)
                attended = attended * qvalid[:, :, None, None].to(dtype)
            output_chunks.append(self.rotary(attended.to(q.dtype), pos, inverse=True))
            if return_aux and self.indexer is not None:
                selected_scores = self.indexer.selected_scores(x[:, begin:end], index_keys, ids,
                    query_latent=latent[:, begin:end], query_positions=pos)
                teacher = probabilities[..., self.window_size:]
                # Normalize by active queries over ALL chunks, not mean of
                # chunk means (which would overweight a short/empty last chunk).
                active = ((teacher.detach().sum((-2, -1)) > 0) & cmask.any(-1)).to(dtype).sum()
                loss_chunks.append(indexer_kl_loss(selected_scores, teacher, cmask) * active)
                active_chunks.append(active)
        attended = torch.cat(output_chunks, 1)
        groups = attended.reshape(b, t, self.output_groups, -1)
        projected = torch.cat([layer(groups[:, :, i]) for i, layer in enumerate(self.output_down)], -1)
        output = self.output_up(projected) * query_valid.unsqueeze(-1).to(projected.dtype)
        if loss_chunks:
            count = torch.stack(active_chunks).sum()
            loss = torch.stack(loss_chunks).sum() / torch.where(count > 0, count, torch.ones_like(count))
        else:
            # HCA has no indexer. Do not attach a synthetic zero gradient to
            # the main model: an auxiliary-only warm-up must not update its
            # optimizer moments or apply decoupled weight decay.
            loss = torch.zeros((), device=output.device, dtype=work_dtype(output))
        return AttentionOutput(output, loss)

    def forward(self, x, *, valid_mask=None, return_aux=False, indexer_warmup=False):
        """Full-sequence training; indexer_warmup attends all visible compressed KV.

        For indexer-only warm-up, backpropagate indexer_loss alone. Main features
        and attention targets are detached; HCA contributes a disconnected zero.
        Warm-up retains the dense compressed attention cost and is not sparse.
        """
        if type(indexer_warmup) is not bool:
            raise TypeError('indexer_warmup must be bool')
        valid = self._validate(x, valid_mask)
        pos = positions(x.shape[1], x.device)
        latent, q, local = self._project(x, pos)
        compressed, comp_valid = self.compressor(x, valid)
        block_pos = positions(compressed.shape[1], x.device) * self.ratio
        compressed = self.rotary(compressed, block_pos)
        index_keys = None
        if self.index_compressor is not None:
            index_keys, _ = self.index_compressor(x.detach(), valid)
            index_keys = self.indexer.rotary(index_keys, block_pos)
        result = self._attend(x, latent, q, pos, valid, local, valid, 0, compressed, comp_valid,
                              index_keys, return_aux=return_aux, indexer_warmup=indexer_warmup)
        return result if return_aux else result.output

    def forward_cached(self, x, cache: CompressedAttentionCache | None = None, *, valid_mask=None):
        """Prefill/decode arbitrary-sized chunks; exact same causal visibility.

        Must be called in eval + no_grad/inference_mode. Rejects another layer's
        cache and parameter changes. Caches store compressed history, a local
        window and bounded raw tails, not the full uncompressed history.
        """
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("cached attention requires eval() and no_grad()/inference_mode()")
        valid = self._validate(x, valid_mask)
        versions = tuple((id(p), p._version) for p in self.parameters())
        if cache is not None:
            if cache.owner != id(self) or cache.parameter_versions != versions:
                raise ValueError("cache belongs to another layer or parameters changed; reset it")
            if cache.local.shape[0] != x.shape[0] or cache.local.device != x.device or cache.local.dtype != x.dtype:
                raise ValueError("attention cache batch/device/dtype mismatch")
        start = 0 if cache is None else cache.seen
        pos = positions(x.shape[1], x.device, start)
        latent, q, local_new = self._project(x, pos)
        new_comp, new_valid, compression = self.compressor.append(x, valid, None if cache is None else cache.compression)
        first_block = 0 if cache is None else cache.compressed.shape[1]
        block_pos = positions(new_comp.shape[1], x.device, first_block) * self.ratio
        new_comp = self.rotary(new_comp, block_pos)
        compressed = new_comp if cache is None else torch.cat((cache.compressed, new_comp), 1)
        comp_valid = new_valid if cache is None else torch.cat((cache.compressed_valid, new_valid), 1)
        index_keys, index_state = None, None
        if self.index_compressor is not None:
            new_index, _, index_state = self.index_compressor.append(x, valid, None if cache is None else cache.index_compression)
            new_index = self.indexer.rotary(new_index, block_pos)
            index_keys = new_index if cache is None else torch.cat((cache.index_keys, new_index), 1)
        local = local_new if cache is None else torch.cat((cache.local, local_new), 1)
        local_valid = valid if cache is None else torch.cat((cache.local_valid, valid), 1)
        local_start = start if cache is None else start - cache.local.shape[1]
        result = self._attend(x, latent, q, pos, valid, local, local_valid, local_start,
            compressed, comp_valid, index_keys, return_aux=False)
        keep = min(self.window_size - 1, local.shape[1])
        # Explicit slicing handles keep=0; [-0:] would retain the whole prompt.
        last = local.shape[1] - keep
        new_cache = CompressedAttentionCache(id(self), versions, start + x.shape[1],
            compressed.detach(), comp_valid.detach(), None if index_keys is None else index_keys.detach(),
            local[:, last:].detach().clone(), local_valid[:, last:].detach().clone(), compression, index_state)
        return result.output, new_cache


class CompressedSparseAttention(_CompressedAttention):
    """CSA: overlapping learned compression + DSA selection + local attention."""
    def __init__(self, width: int, num_heads: int, **kwargs):
        super().__init__(width, num_heads, sparse=True, **kwargs)


class HeavilyCompressedAttention(_CompressedAttention):
    """HCA: non-overlapping compression + dense compressed/local attention."""
    def __init__(self, width: int, num_heads: int, *, compress_ratio=128, **kwargs):
        super().__init__(width, num_heads, sparse=False, compress_ratio=compress_ratio, **kwargs)


CSA = CompressedSparseAttention
HCA = HeavilyCompressedAttention
