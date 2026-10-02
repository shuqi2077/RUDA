"""Device-preserving building blocks for trainable attention and hyper-connections.

All tensor computation stays on the input device. The selection routine is a
portable tournament, not a fused/high-performance top-k kernel. Keeping it here
avoids depending on an unregistered aten::topk on PrivateUse1.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
import torch


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_float(value: float, name: str) -> float:
    if isinstance(value, (bool, torch.Tensor)):
        raise TypeError(f"{name} must be a Python number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def work_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("expected a floating FP16/BF16/FP32/FP64 tensor")
    if x.layout != torch.strided:
        raise ValueError("expected a strided tensor")
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def precision_context(x: torch.Tensor):
    # The RUDA package registers an autocast device module before these APIs run.
    return (torch.autocast(device_type=x.device.type, enabled=False)
            if x.device.type in ("cpu", "cuda", "ruda") else nullcontext())


def positions(length: int, device: torch.device, start: int = 0) -> torch.Tensor:
    # cumsum is already registered for RUDA; arange is not required. Only shape
    # metadata, never tensor values, is read by the Python code.
    return torch.ones(length, device=device, dtype=torch.int64).cumsum(0) + (start - 1)


def rms(x: torch.Tensor, eps: float, weight: torch.Tensor | None = None) -> torch.Tensor:
    dtype = work_dtype(x)
    with precision_context(x):
        value = x.to(dtype)
        result = value * (value.square().mean(-1, keepdim=True) + eps).rsqrt()
        if weight is not None:
            result = result * weight.to(dtype)
    return result.to(x.dtype)


def masked_softmax(scores: torch.Tensor, allowed: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Masked softmax with exactly zero outputs/gradients for all-masked rows."""
    if allowed.dtype != torch.bool or allowed.device != scores.device:
        raise ValueError("allowed must be a boolean mask on the scores' device")
    if scores.shape[dim] == 0:
        return scores * 0
    allowed = allowed.expand_as(scores)
    any_valid = allowed.any(dim=dim, keepdim=True)
    safe = torch.where(any_valid, scores.masked_fill(~allowed, float("-inf")),
                       torch.zeros_like(scores))
    return torch.softmax(safe, dim=dim) * allowed.to(scores.dtype)


def gather_entries(entries: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather [B,S,D] entries with [B,T,K] indices; -1 is a zero sentinel.

    Flattened index_select avoids backward allocating a [B,T,S,D] expanded
    gather gradient. The indexer, not a caller-supplied unsafe index array,
    guarantees upper bounds. PyTorch/RUDA's usual bounds checks remain enabled.
    """
    b, s, d = entries.shape
    if indices.ndim != 3 or indices.shape[0] != b:
        raise ValueError("indices must have shape [batch, queries, selections]")
    if indices.dtype != torch.int64 or indices.device != entries.device:
        raise ValueError("indices must be int64 on the entries' device")
    if s == 0:
        return entries.new_zeros((*indices.shape, d)) + entries.sum() * 0
    valid = indices >= 0
    safe = torch.where(valid, indices, torch.zeros_like(indices))
    offsets = positions(b, entries.device).reshape(b, 1, 1) * s
    selected = entries.reshape(b * s, d).index_select(0, (safe + offsets).reshape(-1))
    selected = selected.reshape(*indices.shape, d)
    return selected * valid.unsqueeze(-1).to(selected.dtype)


def _winner(values: torch.Tensor, indices: torch.Tensor,
            valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce the last axis. Equal scores prefer the smallest external index."""
    while values.shape[-1] > 1:
        size = values.shape[-1]
        pairs = size // 2
        left, right = values[..., :2 * pairs:2], values[..., 1:2 * pairs:2]
        il, ir = indices[..., :2 * pairs:2], indices[..., 1:2 * pairs:2]
        vl, vr = valid[..., :2 * pairs:2], valid[..., 1:2 * pairs:2]
        choose_left = vl & (~vr | (left > right) | ((left == right) & (il <= ir)))
        v = torch.where(choose_left, left, right)
        i = torch.where(choose_left, il, ir)
        ok = vl | vr
        if size % 2:
            v = torch.cat((v, values[..., -1:]), -1)
            i = torch.cat((i, indices[..., -1:]), -1)
            ok = torch.cat((ok, valid[..., -1:]), -1)
        values, indices, valid = v, i, ok
    return values, indices, valid


def stable_topk(scores: torch.Tensor, k: int, *, indices: torch.Tensor | None = None,
                valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-score top-k on the last axis, with deterministic index tie breaks.

    Returns at most min(k,S) slots, sorted by descending score. Invalid slots
    have index -1 and score -inf. Masked -inf is supported; NaNs are outside the
    contract. Runtime is O(k*S), with O(S) live selection workspace per row.
    No sort, topk, host numerical fallback, or scalar device readback is used.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    work_dtype(scores)
    if scores.ndim < 1:
        raise ValueError("scores must have a last dimension")
    n = scores.shape[-1]
    count = min(k, n)
    if indices is None:
        indices = positions(n, scores.device).expand(scores.shape)
    if indices.shape != scores.shape or indices.dtype != torch.int64 or indices.device != scores.device:
        raise ValueError("indices must match scores and use same-device int64 storage")
    if valid is None:
        valid = torch.ones_like(scores, dtype=torch.bool)
    if valid.shape != scores.shape or valid.dtype != torch.bool or valid.device != scores.device:
        raise ValueError("valid must be a same-shape, same-device boolean mask")
    if count == 0:
        return scores[..., :0], indices[..., :0]
    remaining = valid
    all_values, all_indices = [], []
    for _ in range(count):
        value, index, ok = _winner(scores, indices, remaining)
        all_values.append(torch.where(ok, value, torch.full_like(value, float("-inf"))))
        all_indices.append(torch.where(ok, index, torch.full_like(index, -1)))
        remaining = remaining & ~(ok & (indices == index))
    return torch.cat(all_values, -1), torch.cat(all_indices, -1)


def maximum_abs(x: torch.Tensor) -> torch.Tensor:
    """Same-device maximum absolute value without an aten::amax dependency."""
    values = x.abs().reshape(1, -1)
    if not values.shape[-1]:
        return x.new_zeros(())
    ids = positions(values.shape[-1], x.device).reshape(1, -1)
    result, _, _ = _winner(values, ids, torch.ones_like(ids, dtype=torch.bool))
    return result.reshape(())
