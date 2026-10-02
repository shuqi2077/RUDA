"""Learnable integer fake quantization for QAT (not packed INT4 inference).

Forward rounds/clips and dequantizes. The first-order surrogate clips the input
STE and learns the scale from q - x/scale inside the representable range and q
outside. There is no LSQ gradient normalization or automatic calibration.
"""
from __future__ import annotations
import math
from collections.abc import Sequence
import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable


class _LearnedStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, scale: Tensor, lower: int, upper: int):
        normalized = x / scale
        clipped = normalized.clamp(lower, upper)
        # Half away from zero, matching RUDA's integer rounding convention.
        codes = (clipped.sign() * (clipped.abs() + .5).floor()).clamp(lower, upper)
        ctx.save_for_backward(normalized, codes)
        ctx.lower, ctx.upper, ctx.scale_shape = lower, upper, scale.shape
        return codes * scale

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        normalized, codes = ctx.saved_tensors
        inside = (normalized >= ctx.lower) & (normalized <= ctx.upper)
        input_grad = torch.where(inside, grad_output, 0.)
        # where avoids 0*inf at saturation when x/scale overflows.
        scale_factor = torch.where(inside, codes - normalized, codes)
        scale_grad = (grad_output * scale_factor).sum_to_size(ctx.scale_shape)
        return input_grad, scale_grad, None, None


def learned_fake_quantize(x: Tensor, scales: Tensor, *, bits: int = 4,
                          block_shape: Sequence[int] | None = None,
                          symmetric: bool = True) -> Tensor:
    """Fake quantize with trainable FP32 scales on the input's current device.

    `block_shape=None` uses one scale for the entire tensor. Otherwise one scale
    is required for each block, including partial edge blocks; block_shape has
    one positive size per input axis. Symmetric uses [-qmax, qmax]; False uses
    the full signed integer range. Gradients below the 1e-8 scale floor are zero.
    Only first-order derivatives are supported. Inputs/scales must be finite.
    This function returns a FLOAT tensor; it makes no inference-memory claim.
    """
    if not isinstance(x, Tensor) or not isinstance(scales, Tensor):
        raise TypeError("x and scales must be tensors")
    if type(bits) is not int or bits not in (2, 4, 8):
        raise ValueError("bits must be 2, 4, or 8")
    if type(symmetric) is not bool:
        raise TypeError("symmetric must be bool")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("input must be FP16, BF16, or FP32")
    if scales.dtype != torch.float32 or scales.device != x.device:
        raise ValueError("scales must be FP32 on the input device")
    if x.ndim == 0 or x.numel() == 0:
        raise ValueError("input must have nonempty dimensions")
    positive = scales.clamp_min(1e-8)
    if block_shape is None:
        if scales.numel() != 1:
            raise ValueError("per-tensor quantization requires exactly one scale")
        expanded = positive.reshape([1] * x.ndim)
    else:
        block = tuple(block_shape)
        if len(block) != x.ndim or any(type(b) is not int or b <= 0 for b in block):
            raise ValueError("block_shape requires one positive integer per input dimension")
        params = tuple((d + b - 1) // b for d, b in zip(x.shape, block))
        if scales.numel() != math.prod(params):
            raise ValueError("scale count does not match the number of blocks")
        source = tuple(v for p in params for v in (p, 1))
        target = tuple(v for p, b in zip(params, block) for v in (p, b))
        padded = tuple(p * b for p, b in zip(params, block))
        expanded = positive.reshape(source).expand(target).reshape(padded)
        expanded = expanded[tuple(slice(0, d) for d in x.shape)]
    upper = (1 << (bits - 1)) - 1
    lower = -upper if symmetric else -(1 << (bits - 1))
    return _LearnedStep.apply(x.float(), expanded, lower, upper).to(x.dtype)


class LearnedFakeQuantize(nn.Module):
    """Reusable fake-quantization module with a checkpointable scale Parameter.

    Pass an initial FP32 scale tensor of the required per-tensor/per-block size.
    Move the module and model to the same device; keep scales in FP32.
    """
    def __init__(self, initial_scales: Tensor, *, bits: int = 4,
                 block_shape: Sequence[int] | None = None, symmetric: bool = True):
        super().__init__()
        if not isinstance(initial_scales, Tensor) or initial_scales.dtype != torch.float32:
            raise ValueError("initial_scales must be an FP32 tensor")
        if initial_scales.numel() == 0:
            raise ValueError("initial_scales must not be empty")
        if type(bits) is not int or bits not in (2,4,8):
            raise ValueError("bits must be 2, 4, or 8")
        self.scales = nn.Parameter(initial_scales.detach().clone())
        self.bits = bits
        self.block_shape = None if block_shape is None else tuple(block_shape)
        self.symmetric = symmetric

    def forward(self, x: Tensor) -> Tensor:
        return learned_fake_quantize(x, self.scales, bits=self.bits,
            block_shape=self.block_shape, symmetric=self.symmetric)
