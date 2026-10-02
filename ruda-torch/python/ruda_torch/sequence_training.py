"""Model-independent triangular solves and checkpointed gated-delta training."""
from __future__ import annotations

import math
import torch
from torch.autograd.function import once_differentiable
from torch.utils.checkpoint import checkpoint
from ._architecture_ops import positions, precision_context, work_dtype, positive_int


def triangular_mask(x, diagonal=0, *, upper=False):
    if x.ndim < 2:
        raise ValueError('triangular masking requires matrices')
    row = positions(x.shape[-2], x.device).unsqueeze(-1)
    col = positions(x.shape[-1], x.device).unsqueeze(0)
    keep = col - row >= diagonal if upper else col - row <= diagonal
    return torch.where(keep, x, torch.zeros_like(x))


def _solve(a, b, upper, unitriangular):
    if a.device.type != 'ruda':
        return torch.linalg.solve_triangular(a, b, upper=upper, unitriangular=unitriangular)
    from . import _sequence_available
    if not _sequence_available:
        raise RuntimeError('RUDA sequence API 1 is required; rebuild both native libraries')
    shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    aa = a.expand(*shape, *a.shape[-2:]).contiguous()
    bb = b.expand(*shape, *b.shape[-2:]).contiguous()
    return torch.ops.ruda.triangular_solve(aa, bb, bool(upper), bool(unitriangular))


class _TriangularSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, upper, unitriangular):
        result = _solve(a, b, upper, unitriangular)
        ctx.save_for_backward(a, result)
        ctx.options = upper, unitriangular, b.shape
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        a, result = ctx.saved_tensors
        upper, unitriangular, bshape = ctx.options
        db = _solve(a.transpose(-1, -2), gradient, not upper, unitriangular)
        da = triangular_mask(-(db @ result.transpose(-1, -2)),
                             1 if upper and unitriangular else -1 if unitriangular else 0,
                             upper=upper)
        return da.sum_to_size(a.shape), db.sum_to_size(bshape), None, None


def solve_triangular(a, b, *, upper=False, left=True, unitriangular=False):
    """First-order batched triangular solve, including broadcast and right solves.

    RUDA uses its native same-device solve. CPU/CUDA are explicit reference
    devices, not recovery paths for a failed RUDA operation.
    """
    if any(type(flag) is not bool for flag in (upper, left, unitriangular)):
        raise TypeError('triangular solve flags must be Python bools')
    if a.ndim < 2 or b.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError('A must be square and B must be a matrix')
    if a.device != b.device or a.dtype != b.dtype or not a.is_floating_point():
        raise ValueError('solve operands must share a floating dtype and device')
    if b.shape[-2 if left else -1] != a.shape[-1]:
        raise ValueError('incompatible triangular solve dimensions')
    if not left:
        return _TriangularSolve.apply(a.transpose(-1, -2), b.transpose(-1, -2),
                                      not upper, unitriangular).transpose(-1, -2)
    return _TriangularSolve.apply(a, b, upper, unitriangular)


def _delta_chunk(q, k, v, beta, decay, state, scale):
    length = q.shape[-2]
    index = positions(length, q.device)
    lower = index.unsqueeze(-1) >= index.unsqueeze(0)
    strict = index.unsqueeze(-1) > index.unsqueeze(0)
    diagonal = index.unsqueeze(-1) == index.unsqueeze(0)
    cumulative = decay.cumsum(-1)
    difference = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
    # Future positions are masked before exp: negative decays can otherwise
    # overflow in the unused upper triangle and poison its gradients.
    factors = torch.where(lower, difference, torch.zeros_like(difference)).exp()
    kb = k * beta.unsqueeze(-1)
    system = torch.where(strict, (kb @ k.transpose(-1, -2)) * factors,
                         torch.zeros_like(factors)) + diagonal.to(q.dtype)
    rhs = torch.cat((v * beta.unsqueeze(-1), kb * cumulative.exp().unsqueeze(-1)), -1)
    solved = solve_triangular(system, rhs, unitriangular=True)
    u, w = solved[..., :v.shape[-1]], solved[..., v.shape[-1]:]
    residual = u - w @ state
    attention = torch.where(lower, (q * scale @ k.transpose(-1, -2)) * factors,
                            torch.zeros_like(factors))
    output = (q * (cumulative.exp() * scale).unsqueeze(-1)) @ state + attention @ residual
    tail = (cumulative[..., -1:] - cumulative).exp()
    final = state * cumulative[..., -1, None, None].exp() + (k * tail.unsqueeze(-1)).transpose(-1, -2) @ residual
    return output, final


def _delta_scan(q, k, v, beta, decay, state, scale, chunk_size, checkpoint_chunks):
    output = []
    for start in range(0, q.shape[-2], chunk_size):
        end = min(start + chunk_size, q.shape[-2])
        args = (q[..., start:end, :], k[..., start:end, :], v[..., start:end, :],
                beta[..., start:end], decay[..., start:end], state, scale)
        if checkpoint_chunks and torch.is_grad_enabled():
            y, state = checkpoint(_delta_chunk, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            y, state = _delta_chunk(*args)
        output.append(y)
    return torch.cat(output, -2), state


class _NativeDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, decay, initial, scale, chunk_size, checkpoint_chunks):
        ctx.save_for_backward(q, k, v, beta, decay, initial)
        ctx.options = scale, chunk_size, checkpoint_chunks
        return torch.ops.ruda.delta_forward(q.contiguous(), k.contiguous(), v.contiguous(),
                                           beta.contiguous(), decay.contiguous(), initial.contiguous(), scale, chunk_size)

    @staticmethod
    @once_differentiable
    def backward(ctx, gy, gs):
        with torch.enable_grad():
            inputs = [tensor.detach().requires_grad_(needed) for tensor, needed in
                      zip(ctx.saved_tensors, ctx.needs_input_grad[:6])]
            output, final = _delta_scan(*inputs, *ctx.options[:2], ctx.options[2])
            active = [tensor for tensor in inputs if tensor.requires_grad]
            pairs = [(tensor, gradient) for tensor, gradient in ((output, gy), (final, gs))
                     if tensor.requires_grad]
            gradients = torch.autograd.grad(tuple(tensor for tensor, _ in pairs), active,
                         tuple(torch.zeros_like(tensor) if gradient is None else gradient
                               for tensor, gradient in pairs), allow_unused=True)
            iterator = iter(gradients)
            return tuple(next(iterator) if tensor.requires_grad else None for tensor in inputs) + (None,) * 3


def gated_delta_rule(query, key, value, beta, log_decay, *, initial_state=None,
                     query_scale=None, normalize_qk=False, norm_eps=1e-6,
                     chunk_size=64, checkpoint_chunks=True, native_forward=True):
    """Differentiable gated delta recurrence with [B,H,T,D] tensor layout.

    S_t = exp(g_t) S_(t-1) + k_t outer
          (beta_t * (v_t - k_t @ (exp(g_t) S_(t-1))))
    y_t = query_scale * q_t @ S_t.

    RUDA reuses ruDNN's native chunk forward by default. The differentiable
    chunk algorithm uses FP32 state/intermediates (FP64 for explicit CPU FP64
    inputs), device GEMMs and a native triangular solve on RUDA.
    Backward recomputes chunk intermediates through the same-device operators;
    this is not a standalone fused DeltaNet backward kernel. No state is
    detached between chunks. Both output and final-state gradients propagate.
    """
    positive_int(chunk_size, 'chunk_size')
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4 or value.shape[:3] != query.shape[:3]:
        raise ValueError('query/key/value must be matching [B,H,T,D] tensors')
    if min(query.shape[0], query.shape[1], query.shape[-1], value.shape[-1]) < 1:
        raise ValueError('batch, heads and feature dimensions must be positive')
    if beta.shape != query.shape[:3] or log_decay.shape != beta.shape:
        raise ValueError('beta/log_decay must have shape [B,H,T]')
    tensors = (key, value, beta, log_decay)
    if any(t.device != query.device or not t.is_floating_point() for t in tensors):
        raise ValueError('all operands must be floating tensors on one device')
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError('query/key/value storage dtypes must match')
    dtype = work_dtype(query)
    if isinstance(query_scale, (bool, torch.Tensor)) or isinstance(norm_eps, (bool, torch.Tensor)):
        raise TypeError('query_scale and norm_eps must be Python numbers')
    scale = query.shape[-1] ** -.5 if query_scale is None else float(query_scale)
    if not math.isfinite(scale) or not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError('invalid query scale or normalization epsilon')
    shape = (*query.shape[:2], query.shape[-1], value.shape[-1])
    if initial_state is not None and (initial_state.shape != shape or initial_state.device != query.device or not initial_state.is_floating_point()):
        raise ValueError('initial state must have shape [B,H,K,V] on the input device')
    with precision_context(query):
        q, k, v = (t.to(dtype) for t in (query, key, value))
        if normalize_qk:
            q = q * (q.square().sum(-1, keepdim=True) + norm_eps).rsqrt()
            k = k * (k.square().sum(-1, keepdim=True) + norm_eps).rsqrt()
        state = torch.zeros(shape, device=query.device, dtype=dtype) if initial_state is None else initial_state.to(dtype)
        if not query.shape[-2]:
            # Preserve the connected empty-output and identity-state contract.
            return value * 0, state
        args = q, k, v, beta.to(dtype), log_decay.to(dtype), state, scale, chunk_size, checkpoint_chunks
        if query.device.type == 'ruda' and native_forward:
            from . import _sequence_available
            if not _sequence_available:
                raise RuntimeError('RUDA sequence API 1 required for native DeltaNet; rebuild both libraries')
            output, state = _NativeDelta.apply(*args)
        else:
            output, state = _delta_scan(*args)
        return output.to(value.dtype), state
