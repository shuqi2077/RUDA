"""Continuous GPU router weights for model-selected expert indices.

No selection/grouping, CPU fallback or correction-bias gradient is introduced.
Scoring and reductions use FP32; output is FP32, dlogits matches logits storage.
"""
from __future__ import annotations

import math
import struct
import torch
from . import _C


def _options(scoring: str, renormalize: bool, scale: float) -> tuple[int, bool, float]:
    if not isinstance(scoring, str) or scoring not in ("softmax", "sigmoid"):
        raise ValueError("scoring must be 'softmax' or 'sigmoid'")
    if type(renormalize) is not bool:
        raise TypeError("renormalize must be a bool")
    if isinstance(scale, (bool, torch.Tensor)):
        raise TypeError("scale must be a Python scalar, not a bool or tensor")
    try:
        value = struct.unpack("f", struct.pack("f", float(scale)))[0]
    except (ValueError, TypeError, OverflowError, struct.error) as error:
        raise ValueError("scale must be representable as finite positive FP32") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("scale must be finite positive FP32")
    return int(scoring == "sigmoid"), renormalize, value


class _SelectedRouterWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, indices, scoring, renormalize, scale):
        ctx.save_for_backward(logits, indices)
        ctx.options = (scoring, renormalize, scale)
        ctx.set_materialize_grads(False)
        return _C.router_weights_forward(logits, indices, *ctx.options)

    @staticmethod
    def backward(ctx, gradient):
        if torch.is_grad_enabled():
            raise RuntimeError("RUDA router weights support first-order gradients only")
        if gradient is None:
            return None, None, None, None, None
        logits, indices = ctx.saved_tensors  # Also checks in-place version changes.
        dx = _C.router_weights_backward(logits, indices, gradient.contiguous(), *ctx.options)
        return dx, None, None, None, None


def selected_router_weights(logits: torch.Tensor, indices: torch.Tensor, *,
                            scoring: str = "softmax", renormalize: bool = False,
                            scale: float = 1.0) -> torch.Tensor:
    """FP32 weights and first-order dlogits for a fixed expert selection.

    logits: contiguous ruda [tokens,experts], FP32/FP16/BF16.
    indices: contiguous ruda [tokens,top_k], int32/int64, 1<=top_k<=min(E,64).
    Softmax is over ALL experts before gathering; sigmoid is pointwise.
    Renormalization, when enabled, is over the gathered slots; scale is applied
    last. Repeated indices have gather semantics and accumulate in backward.

    There is no host index-value synchronization: invalid indices return NaN
    for the entire affected row (forward AND backward), with bounds-safe reads.
    NaN/Inf or a zero selected denominator are never replaced with uniform
    weights. This is not an auxiliary/load-balancing loss or a model router.
    """
    options = _options(scoring, renormalize, scale)
    for t, name in ((logits, "logits"), (indices, "indices")):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if t.device.type != "ruda" or t.device.index not in (None, 0):
            raise ValueError(f"{name} requires native ruda:0 storage")
        if not t.is_contiguous() or t.is_conj() or t.is_neg():
            raise ValueError(f"{name} must be contiguous and resolved")
    if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("router logits must be FP32, FP16 or BF16")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("router indices must be int32 or int64")
    if (logits.ndim != 2 or indices.ndim != 2 or logits.shape[0] != indices.shape[0]
            or logits.shape[1] == 0 or not 1 <= indices.shape[1] <= min(64, logits.shape[1])
            or logits.device != indices.device):
        raise ValueError("invalid router shapes or device")
    if not hasattr(_C, "router_weights_forward"):
        raise RuntimeError("RUDA router API 1 unavailable; rebuild Rust and C++ extensions")
    return _SelectedRouterWeights.apply(logits, indices, *options)
