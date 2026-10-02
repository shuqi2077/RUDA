"""Trainable manifold-constrained hyper-connections (mHC).

Implements Eqs. 7--9 of arXiv:2512.24880, using log-space Sinkhorn updates.
This is a differentiable composition of tensor operations, not a fused mHC
kernel. A branch passed to MHCResidual must NOT add its own residual input.
"""
from __future__ import annotations

import math
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ._architecture_ops import positive_int, positive_float, work_dtype, precision_context


class MHCCoefficients(NamedTuple):
    pre: torch.Tensor
    post: torch.Tensor
    residual: torch.Tensor


def sinkhorn(logits: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """Positive row/column normalization (columns first, then rows).

    Row sums are one within rounding. Column sums approach one as iterations
    increase; a finite iteration count is NOT an exact manifold projection.
    Accumulation is FP32 (FP64 for FP64 input), including under autocast.
    """
    positive_int(iterations, "iterations")
    if logits.ndim < 2 or logits.shape[-1] != logits.shape[-2] or not logits.shape[-1]:
        raise ValueError("Sinkhorn requires nonempty square trailing dimensions")
    dtype = work_dtype(logits)
    with precision_context(logits):
        log_matrix = logits.to(dtype)
        for _ in range(iterations):
            log_matrix = F.log_softmax(log_matrix, dim=-2)
            log_matrix = F.log_softmax(log_matrix, dim=-1)
        return log_matrix.exp()


class MHC(nn.Module):
    """Generate and apply dynamic mHC mappings for [..., streams, width] states.

    Parameters are registered normally and participate in optimizers,
    state_dict, autograd, checkpointing, and ruda_torch.compile. FP16/BF16
    activations use FP32 mapping arithmetic; FP64 is retained for diagnostics.
    """
    def __init__(self, width: int, streams: int = 4, *, sinkhorn_iterations: int = 20,
                 eps: float = 1e-6, gate_init: float = 0.01, device=None, dtype=None):
        super().__init__()
        self.width = positive_int(width, "width")
        self.streams = positive_int(streams, "streams")
        self.sinkhorn_iterations = positive_int(sinkhorn_iterations, "sinkhorn_iterations")
        self.eps = positive_float(eps, "eps")
        if not math.isfinite(gate_init):
            raise ValueError("gate_init must be finite")
        n = self.streams
        options = dict(device=device, dtype=dtype or torch.float32)
        self.mapping = nn.Parameter(torch.empty(n * width, n * n + 2 * n, **options))
        self.alpha = nn.Parameter(torch.full((3,), gate_init, **options))
        self.bias = nn.Parameter(torch.zeros(n * n + 2 * n, **options))
        nn.init.normal_(self.mapping, std=(n * width) ** -0.5)
        # Start close to a residual-preserving topology; these are explicit
        # initialization choices, not a claim to reproduce a V4 checkpoint.
        with torch.no_grad():
            if n > 1:
                self.bias[:n].fill_(-math.log(n - 1))  # sigmoid = 1/n
            residual = self.bias[2 * n:].view(n, n)
            for i in range(n):
                residual[i, i] = 2.0

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.width:
            raise ValueError("input must end in width")
        return x.unsqueeze(-2).expand(*x.shape[:-1], self.streams, self.width).clone()

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        return x.mean(dim=-2)

    def _check(self, x: torch.Tensor):
        work_dtype(x)
        if x.ndim < 2 or tuple(x.shape[-2:]) != (self.streams, self.width):
            raise ValueError("mHC state must have shape [..., streams, width]")
        if x.device != self.mapping.device:
            raise ValueError("mHC state and parameters must be on the same device")

    def coefficients(self, x: torch.Tensor) -> MHCCoefficients:
        self._check(x)
        n, dtype = self.streams, work_dtype(x)
        with precision_context(x):
            flat = x.to(dtype).flatten(-2)
            normalized = flat * (flat.square().mean(-1, keepdim=True) + self.eps).rsqrt()
            projected = normalized @ self.mapping.to(dtype)
            pre = torch.sigmoid(projected[..., :n] * self.alpha[0].to(dtype) + self.bias[:n].to(dtype))
            post = 2 * torch.sigmoid(projected[..., n:2*n] * self.alpha[1].to(dtype) + self.bias[n:2*n].to(dtype))
            raw = (projected[..., 2*n:] * self.alpha[2].to(dtype) + self.bias[2*n:].to(dtype))
            residual = sinkhorn(raw.reshape(*x.shape[:-2], n, n), self.sinkhorn_iterations)
        return MHCCoefficients(pre, post, residual)

    def pre(self, x: torch.Tensor) -> tuple[torch.Tensor, MHCCoefficients]:
        coefficients = self.coefficients(x)
        with precision_context(x):
            merged = (x.to(coefficients.pre.dtype) * coefficients.pre.unsqueeze(-1)).sum(-2)
        return merged.to(x.dtype), coefficients

    def post(self, x: torch.Tensor, branch_output: torch.Tensor,
             coefficients: MHCCoefficients) -> torch.Tensor:
        self._check(x)
        if branch_output.shape != (*x.shape[:-2], self.width):
            raise ValueError("branch output must preserve all dimensions except streams")
        work_dtype(branch_output)
        if branch_output.device != x.device:
            raise ValueError("branch output must preserve the state device")
        with precision_context(x):
            dtype = coefficients.residual.dtype
            residual = coefficients.residual @ x.to(dtype)
            update = coefficients.post.unsqueeze(-1) * branch_output.to(dtype).unsqueeze(-2)
        return (residual + update).to(x.dtype)

    def forward(self, x: torch.Tensor, branch: nn.Module, *args, **kwargs) -> torch.Tensor:
        merged, coefficients = self.pre(x)
        return self.post(x, branch(merged, *args, **kwargs), coefficients)


class MHCResidual(nn.Module):
    """Wrap a residual-free branch; inputs/outputs keep the stream dimension."""
    def __init__(self, branch: nn.Module, width: int, streams: int = 4, *,
                 checkpoint_branch: bool = False, **kwargs):
        super().__init__()
        if not isinstance(branch, nn.Module):
            raise TypeError("branch must be an nn.Module without its own residual addition")
        self.branch = branch
        self.connection = MHC(width, streams, **kwargs)
        self.checkpoint_branch = bool(checkpoint_branch)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        merged, coefficients = self.connection.pre(x)
        if self.checkpoint_branch and self.training and torch.is_grad_enabled():
            output = checkpoint(self.branch, merged, *args, use_reentrant=False, **kwargs)
        else:
            output = self.branch(merged, *args, **kwargs)
        return self.connection.post(x, output, coefficients)


class MHCSequential(nn.Module):
    """Expand once, run residual-free branches with mHC, then average streams."""
    def __init__(self, width: int, branches, streams: int = 4, **kwargs):
        super().__init__()
        positive_int(width, "width"); positive_int(streams, "streams")
        self.width, self.streams = width, streams
        self.layers = nn.ModuleList([MHCResidual(branch, width, streams, **kwargs) for branch in branches])
        if not self.layers:
            raise ValueError("at least one branch is required")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.layers[0].connection.expand(x)
        for layer in self.layers:
            state = layer(state)
        return self.layers[-1].connection.reduce(state)
