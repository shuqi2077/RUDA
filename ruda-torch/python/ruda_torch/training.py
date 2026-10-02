"""Explicit first-order RUDA training primitives; no CPU numerical fallback.

StaticGraph(training=True) and the general model compiler are opt-in training
paths. These eager primitives do not capture an optimizer step. Use FP32 losses.

v28 adds native first-order LayerNorm training with FP32 saved statistics.
"""
from __future__ import annotations

import math
import struct
from typing import Iterable
import torch

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _native():
    from . import _C, _training_available
    if not _training_available:
        raise RuntimeError("RUDA training API 4 is unavailable; rebuild the Rust and C++ libraries")
    return _C


def _f32(value: float, name: str, *, positive=False, nonnegative=False) -> float:
    if isinstance(value, (bool, torch.Tensor)):
        raise TypeError(f"{name} must be a Python number")
    try:
        number = float(value)
        rounded = struct.unpack('f', struct.pack('f', number))[0]
    except (ValueError, TypeError, OverflowError, struct.error) as error:
        raise ValueError(f"{name} must be finite FP32") from error
    if not math.isfinite(rounded) or (positive and rounded <= 0) or (nonnegative and rounded < 0):
        raise ValueError(f"{name} is outside the supported FP32 range")
    return number


def _tensor(x: torch.Tensor, name: str):
    if not isinstance(x, torch.Tensor) or x.device.type != 'ruda' or x.device.index not in (None, 0):
        raise ValueError(f"{name} must be a ruda:0 tensor; CPU fallback is disabled")
    if x.dtype not in _DTYPES or x.layout != torch.strided or not x.is_contiguous() or x.is_conj() or x.is_neg():
        raise ValueError(f"{name} must be contiguous FP32/FP16/BF16 without unresolved view bits")


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, epsilon):
        y, stats = _native().training_rms_forward(x, weight, epsilon)
        ctx.save_for_backward(x, weight, stats)
        ctx.set_materialize_grads(False)
        return y

    @staticmethod
    def backward(ctx, dy):
        if torch.is_grad_enabled():
            raise RuntimeError("RUDA fused RMSNorm supports first-order gradients only; create_graph is unsupported")
        if dy is None:
            return None, None, None
        x, weight, stats = ctx.saved_tensors
        dx, dw = _native().training_rms_backward(x, weight, dy.contiguous(), stats,
            ctx.needs_input_grad[0], weight is not None and ctx.needs_input_grad[1])
        return dx, dw, None


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, *, eps: float | None = None):
    """Last-axis RMSNorm, input-dtype output, FP32 statistics and optional FP32 weight.

    Saves input, weight and one FP32 statistic per row, not normalized activations.
    Supports first-order backward only; this is not aten's mixed-dtype contract.
    """
    _tensor(x, 'input')
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError('RMSNorm requires a positive last dimension')
    if weight is not None:
        _tensor(weight, 'weight')
        if weight.shape != (x.shape[-1],) or weight.device != x.device or weight.dtype not in (x.dtype, torch.float32):
            raise ValueError('RMSNorm weight must match width/device and be input dtype or FP32')
    epsilon = torch.finfo(x.dtype).eps if eps is None else _f32(eps, 'eps', positive=True)
    return _RMSNorm.apply(x, weight, epsilon)


class RMSNorm(torch.nn.Module):
    """Trainable last-axis normalization. CPU construction is allowed; forward requires RUDA."""
    def __init__(self, width: int, *, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError('width must be a positive integer')
        self.width = width
        self.eps = None if eps is None else _f32(eps, 'eps', positive=True)
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(width, device=device, dtype=dtype or torch.float32))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        if x.ndim == 0 or x.shape[-1] != self.width:
            raise ValueError('RMSNorm input width mismatch')
        return rms_norm(x, self.weight, eps=self.eps)



class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, epsilon):
        y, mean, rstd = _native().training_layer_forward(x, weight, bias, epsilon)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.has_weight = weight is not None
        ctx.has_bias = bias is not None
        ctx.set_materialize_grads(False)
        return y

    @staticmethod
    def backward(ctx, dy):
        if torch.is_grad_enabled():
            raise RuntimeError('RUDA fused LayerNorm supports first-order gradients only; create_graph is unsupported')
        if dy is None:
            return None, None, None, None
        x, weight, bias, mean, rstd = ctx.saved_tensors
        dx, dw, db = _native().training_layer_backward(
            x, weight if ctx.has_weight else None, bias if ctx.has_bias else None,
            dy.contiguous(), mean, rstd,
            ctx.needs_input_grad[0], ctx.has_weight and ctx.needs_input_grad[1],
            ctx.has_bias and ctx.needs_input_grad[2])
        return dx, dw, db, None


def layer_norm(x: torch.Tensor, weight: torch.Tensor | None = None, bias: torch.Tensor | None = None, *, eps: float = 1e-5):
    """Trainable last-axis LayerNorm with FP32 mean/rstd and fused first-order backward."""
    _tensor(x, 'input')
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError('LayerNorm requires a positive last dimension')
    for value, name in ((weight, 'weight'), (bias, 'bias')):
        if value is not None:
            _tensor(value, name)
            if value.shape != (x.shape[-1],) or value.device != x.device or value.dtype not in (x.dtype, torch.float32):
                raise ValueError(f'LayerNorm {name} must match width/device and be input dtype or FP32')
    epsilon = _f32(eps, 'eps', positive=True)
    return _LayerNorm.apply(x, weight, bias, epsilon)


class LayerNorm(torch.nn.Module):
    """Native last-axis LayerNorm for RUDA training; affine parameters default to FP32."""
    def __init__(self, width: int, *, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None):
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError('width must be a positive integer')
        self.width = width
        self.eps = _f32(eps, 'eps', positive=True)
        affine_dtype = dtype or torch.float32
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(width, device=device, dtype=affine_dtype))
            if bias:
                self.bias = torch.nn.Parameter(torch.zeros(width, device=device, dtype=affine_dtype))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        if x.ndim == 0 or x.shape[-1] != self.width:
            raise ValueError('LayerNorm input width mismatch')
        return layer_norm(x, self.weight, self.bias, eps=self.eps)


class _SiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        ctx.set_materialize_grads(False)
        return _native().training_silu_forward(gate, up)

    @staticmethod
    def backward(ctx, dy):
        if torch.is_grad_enabled():
            raise RuntimeError('RUDA fused SiLU-mul supports first-order gradients only')
        if dy is None:
            return None, None
        gate, up = ctx.saved_tensors
        return _native().training_silu_backward(gate, up, dy.contiguous(), *ctx.needs_input_grad)


def silu_mul(gate: torch.Tensor, up: torch.Tensor):
    """Differentiable fused SiLU(gate)*up, preserving low-precision storage boundaries."""
    _tensor(gate, 'gate'); _tensor(up, 'up')
    if gate.shape != up.shape or gate.dtype != up.dtype or gate.device != up.device:
        raise ValueError('SiLU-mul requires matching shapes, dtypes and devices, without broadcasting')
    return _SiluMul.apply(gate, up)


def _no_overlap(tensors: Iterable[torch.Tensor]):
    # Contiguous intervals only. Same allocation, disjoint views are allowed.
    groups = {}
    for t in tensors:
        if not t.numel():
            continue
        key = (t.device, t.untyped_storage()._cdata)
        start = t.storage_offset() * t.element_size()
        groups.setdefault(key, []).append((start, start + t.numel() * t.element_size()))
    for spans in groups.values():
        spans.sort()
        for left, right in zip(spans, spans[1:]):
            if right[0] < left[1]:
                raise ValueError('optimizer parameters, gradients and states must not overlap')


class AdamW(torch.optim.Optimizer):
    """Single-device eager AdamW with FP32 moments and low-precision master weights.

    Dense contiguous parameters only. One fused update per active parameter.
    Each step unscales/checks all gradients on the GPU, then reads one FP32 flag
    (4 bytes) before any parameter/moment/counter is updated. This deliberate
    synchronization is not graph-capturable. An overflow skips the entire step.

    Experimental fused_step=True keeps gradients read-only, fuses scale/clip
    into updates, and submits all updates in one native call. max_grad_norm
    optionally clips the global unscaled L2 norm across ALL parameter groups.
    It is not clip_grad_norm_: .grad tensors remain unchanged. One 12-byte
    statistics readback replaces the old 4-byte flag readback.
    hierarchical_stats=True opt-in reduces large statistics workspaces in parallel
    levels (fan-in 1024). It adds up to two kernels and <=49,200 bytes of scratch.
    Small workspaces (<=1024 rows) retain the single-warp final merge.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
                 *, fused_step=False, max_grad_norm=None, hierarchical_stats=False):
        self.fused_step = fused_step
        self.hierarchical_stats = hierarchical_stats
        self.max_grad_norm = max_grad_norm
        self._validate_step_options()
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self._scratch = None
        self._analysis_scratch = None
        self._reduction_scratch = None
        self.last_grad_norm = None
        self.last_clip_coef = None
        self._poisoned = False
        self.last_step_skipped = False
        self.last_step_had_grad = False
        self._validate_groups()

    def _validate_step_options(self):
        if type(self.fused_step) is not bool:
            raise TypeError('fused_step must be bool')
        if type(self.hierarchical_stats) is not bool:
            raise TypeError('hierarchical_stats must be bool')
        if self.hierarchical_stats and not self.fused_step:
            raise ValueError('hierarchical_stats requires fused_step=True')
        if self.max_grad_norm is not None:
            _f32(self.max_grad_norm, 'max_grad_norm', nonnegative=True)
            if not self.fused_step:
                raise ValueError('max_grad_norm requires the explicit fused_step=True path')

    def _validate_groups(self, groups=None):
        params = []
        for group in self.param_groups if groups is None else groups:
            for unsupported in ('amsgrad', 'maximize', 'differentiable', 'capturable'):
                if group.get(unsupported, False):
                    raise ValueError(f'{unsupported} is not supported by RUDA AdamW')
            _f32(group['lr'], 'lr', nonnegative=True)
            _f32(group['eps'], 'eps', positive=True)
            _f32(group['weight_decay'], 'weight_decay', nonnegative=True)
            if len(group['betas']) != 2:
                raise ValueError('betas must have two elements')
            for beta in group['betas']:
                b = _f32(beta, 'beta', nonnegative=True)
                if struct.unpack('f', struct.pack('f', b))[0] >= 1:
                    raise ValueError('beta must be strictly below one in FP32')
            for p in group['params']:
                _tensor(p, 'parameter')
                if not p.is_leaf or p.is_inference():
                    raise ValueError('optimizer parameters must be normal leaf tensors')
                params.append(p)
        _no_overlap(params)

    @torch.no_grad()
    def step(self, closure=None, *, loss_scale=1.0):
        if self._poisoned:
            raise RuntimeError('a native optimizer call failed; reload a known checkpoint before reusing this optimizer')
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._validate_groups()
        self._validate_step_options()
        scale = _f32(loss_scale, 'loss_scale', positive=True)
        inverse = _f32(1.0 / scale, 'inverse loss scale', positive=True)
        active = []
        occupied = [p for group in self.param_groups for p in group['params']]
        for group in self.param_groups:
            for p in group['params']:
                g = p.grad
                if g is None or not p.numel():
                    continue
                _tensor(g, 'gradient')
                if g.shape != p.shape or g.dtype != p.dtype or g.device != p.device or g.requires_grad:
                    raise ValueError('gradients must match parameters and not require higher-order derivatives')
                active.append((group, p, g))
                occupied.append(g)
                state = self.state.get(p, {})
                if state:
                    self._check_state(p, state)
                    for key in ('exp_avg','exp_avg_sq','master_copy'):
                        if key in state:
                            _tensor(state[key], key)
                            if state[key].device != p.device:
                                raise ValueError('optimizer state device mismatch')
                            occupied.append(state[key])
        _no_overlap(occupied)
        self.last_step_skipped = False
        self.last_grad_norm = None
        self.last_clip_coef = None
        self.last_step_had_grad = bool(active)
        if not active:
            return loss
        if len(active) > 4096:
            raise ValueError('training API 4 accepts at most 4096 active parameter tensors')
        if self.fused_step:
            return self._step_fused(active, inverse, loss)
        count = sum(max(1, min(1024, (g.numel() + 31)//32)) for _, _, g in active)
        device = active[0][1].device
        if self._scratch is None or self._scratch[0].numel() != count or self._scratch[0].device != device:
            self._scratch = (torch.empty(count, device=device, dtype=torch.float32),
                             torch.empty((), device=device, dtype=torch.float32))
        native = _native()
        try:
            native.training_unscale_([g for _,_,g in active], inverse, *self._scratch)
            overflow = float(self._scratch[1].item())
            if overflow not in (0.0, 1.0):
                raise RuntimeError('invalid native nonfinite-gradient flag')
            if overflow:
                self.last_step_skipped = True
                return loss
            # State initialization happens only after the whole optimizer passed
            # the finite check: an initial overflow allocates no moment/master state.
            for group, p, g in active:
                state = self.state[p]
                if not state:
                    state.update(step=0, param_dtype=str(p.dtype),
                        exp_avg=torch.zeros_like(p, dtype=torch.float32),
                        exp_avg_sq=torch.zeros_like(p, dtype=torch.float32))
                    if p.dtype != torch.float32:
                        state['master_copy'] = p.detach().to(torch.float32, copy=True)
                beta1, beta2 = group['betas']
                next_step = state['step'] + 1
                native.training_adamw_(p, g, state.get('master_copy', p), state['exp_avg'], state['exp_avg_sq'],
                    group['lr'], beta1, beta2, group['eps'], group['weight_decay'],
                    1.0-beta1**next_step, 1.0-beta2**next_step)
                state['step'] = next_step
        except Exception:
            # The driver has no transaction rollback. Do not retry a partly
            # unscaled/updated set of gradients with apparently fresh counters.
            self._poisoned = True
            raise
        return loss

    @torch.no_grad()
    def _step_fused(self, active, inverse, loss):
        """Two native calls: read-only analysis, then one batched update command.

        This is not one multi-tensor GPU kernel: each parameter still launches
        its own update. Gradients remain scaled and unchanged, so callers must
        zero them before accumulating the NEXT step. FP32-unscale rounding is
        deliberately not the same as API 1's FP16/BF16 gradient writeback.
        """
        rows = sum(max(1, min(1024, (g.numel() + 31)//32)) for _, _, g in active)
        needed = rows * 3
        device = active[0][1].device
        old = self._analysis_scratch
        if old is None or old[0].device != device or old[0].numel() < needed:
            capacity = needed if old is None or old[0].device != device else min(4096*1024*3, max(needed, old[0].numel()*2))
            self._analysis_scratch = (torch.empty(capacity, device=device, dtype=torch.float32),
                                      torch.empty(3, device=device, dtype=torch.float32))
        storage, report = self._analysis_scratch
        workspace = storage[:needed]
        native = _native()
        try:
            if self.hierarchical_stats:
                from ._gradient_stats import statistics_plan
                plan = statistics_plan(rows)
                needed_reduction = plan.scratch_elements
                old_reduction = self._reduction_scratch
                if old_reduction is None or old_reduction.device != device or old_reduction.numel() < needed_reduction:
                    # The exact bounded plan is <= 49,200 bytes. No geometric
                    # over-allocation; retain its capacity across smaller steps.
                    self._reduction_scratch = torch.empty(needed_reduction, device=device, dtype=torch.float32)
                native.training_analyze_hierarchical([g for _,_,g in active], inverse,
                    workspace, self._reduction_scratch[:needed_reduction], report, self.max_grad_norm is not None)
            else:
                native.training_analyze([g for _,_,g in active], inverse, workspace, report,
                                        self.max_grad_norm is not None)
            # One explicit 12-byte readback gates the ENTIRE optimizer step.
            # This is host scheduling/metadata, not CPU gradient computation.
            bad, magnitude, squares = report.detach().cpu().tolist()
            if bad not in (0.0, 1.0) or not all(math.isfinite(x) and x >= 0 for x in (magnitude, squares)):
                raise RuntimeError('invalid native gradient statistics')
            if bad:
                self.last_step_skipped = True
                return loss
            clip = 1.0
            if self.max_grad_norm is not None:
                if (magnitude == 0.0) != (squares == 0.0):
                    raise RuntimeError('inconsistent native gradient statistics')
                # Host double reconstructs a norm that may exceed FP32 range.
                norm = magnitude * math.sqrt(squares)
                clip = min(1.0, float(self.max_grad_norm) / (norm + 1e-6))
                self.last_grad_norm = norm
            clip = struct.unpack('f', struct.pack('f', clip))[0]
            self.last_clip_coef = clip
            ps, gs, masters, first, second, hypers, steps = [], [], [], [], [], [], []
            # Initialize all state and validate all correction factors BEFORE
            # invoking the first parameter-write kernel.
            for group, p, g in active:
                state = self.state[p]
                if not state:
                    state.update(step=0, param_dtype=str(p.dtype),
                                 exp_avg=torch.zeros_like(p, dtype=torch.float32),
                                 exp_avg_sq=torch.zeros_like(p, dtype=torch.float32))
                    if p.dtype != torch.float32:
                        state['master_copy'] = p.detach().to(torch.float32, copy=True)
                b1, b2 = group['betas']; step = state['step'] + 1
                c1 = _f32(1.0-b1**step, 'correction1', positive=True)
                c2 = _f32(1.0-b2**step, 'correction2', positive=True)
                ps.append(p); gs.append(g); masters.append(state.get('master_copy', p))
                first.append(state['exp_avg']); second.append(state['exp_avg_sq'])
                hypers.append([group['lr'], b1, b2, group['eps'], group['weight_decay'], c1, c2])
                steps.append((state, step))
            native.training_adamw_batch_(ps, gs, masters, first, second, hypers, inverse, clip)
            for state, step in steps:
                state['step'] = step
        except Exception:
            # Preflight is all-or-none; device failures need checkpoint recovery.
            self._poisoned = True
            raise
        return loss

    @staticmethod
    def _check_state(p, state):
        keys = {'step','param_dtype','exp_avg','exp_avg_sq'} | ({'master_copy'} if p.dtype != torch.float32 else set())
        if set(state) != keys or state['param_dtype'] != str(p.dtype):
            raise ValueError('optimizer state format or parameter dtype mismatch')
        if type(state['step']) is not int or state['step'] < 0:
            raise ValueError('optimizer step must be a nonnegative integer')
        for key in keys - {'step','param_dtype'}:
            value = state[key]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.shape != p.shape or not value.is_contiguous():
                raise ValueError(f'{key} must be a matching contiguous FP32 tensor')

    def state_dict(self):
        if self._poisoned:
            raise RuntimeError('cannot checkpoint a failed native optimizer step')
        result = super().state_dict()
        result['ruda_training_version'] = 1
        result['ruda_step_options'] = {'version': 1, 'fused_step': self.fused_step,
                                       'max_grad_norm': self.max_grad_norm}
        if self.hierarchical_stats:
            result['ruda_step_options'].update(version=2, hierarchical_stats=True)
        return result

    def load_state_dict(self, state_dict):
        if self._optimizer_load_state_dict_pre_hooks or self._optimizer_load_state_dict_post_hooks:
            raise RuntimeError('training API 4 does not support optimizer load-state hooks; remove hooks before restoring')
        if state_dict.get('ruda_training_version') != 1:
            raise ValueError('expected RUDA training optimizer checkpoint version 1')
        options = state_dict.get('ruda_step_options', {'version': 1, 'fused_step': False, 'max_grad_norm': None})
        expected_keys = {'version', 'fused_step', 'max_grad_norm'}
        if options.get('version') == 2:
            expected_keys.add('hierarchical_stats')
        if set(options) != expected_keys or options.get('version') not in (1, 2):
            raise ValueError('invalid RUDA optimizer step options')
        hierarchical = options.get('hierarchical_stats', False)
        if type(hierarchical) is not bool or (hierarchical and not options.get('fused_step')):
            raise ValueError('invalid checkpoint hierarchical_stats option')
        if type(options['fused_step']) is not bool:
            raise ValueError('invalid checkpoint fused_step flag')
        if options['max_grad_norm'] is not None:
            _f32(options['max_grad_norm'], 'max_grad_norm', nonnegative=True)
            if not options['fused_step']:
                raise ValueError('checkpoint clipping requires fused_step')
        groups = state_dict.get('param_groups', [])
        if len(groups) != len(self.param_groups) or any(len(a['params']) != len(b['params']) for a,b in zip(groups,self.param_groups)):
            raise ValueError('optimizer checkpoint parameter groups mismatch')
        self._validate_groups([{**saved, 'params': current['params']} for saved, current in zip(groups, self.param_groups)])
        ids = [sid for g in groups for sid in g['params']]
        if len(ids) != len(set(ids)) or not set(state_dict.get('state', {})).issubset(ids):
            raise ValueError('optimizer checkpoint has duplicate or unknown parameter IDs')
        preserved = []
        for saved, current in zip(groups, self.param_groups):
            for sid, p in zip(saved['params'], current['params']):
                state = state_dict['state'].get(sid, {})
                if state:
                    self._check_state(p, state)
                    restored = {k: (v.detach().to(device=p.device, dtype=torch.float32, copy=True)
                                    if isinstance(v, torch.Tensor) else v) for k,v in state.items()}
                    preserved.append((p, restored))
        # PyTorch's generic loader casts optimizer tensors to parameter dtype.
        # Restore from the original FP32 checkpoint, NEVER from that narrowed copy.
        super().load_state_dict({k:v for k,v in state_dict.items() if k not in ('ruda_training_version', 'ruda_step_options')})
        for p, restored in preserved:
            self.state[p] = restored
        self._scratch = None
        self._analysis_scratch = None
        self._reduction_scratch = None
        self.last_grad_norm = None
        self.last_clip_coef = None
        self._poisoned = False
        self.last_step_skipped = False
        self.last_step_had_grad = False
        self.hierarchical_stats = hierarchical
        self.fused_step = options['fused_step']
        self.max_grad_norm = options['max_grad_norm']
        self._validate_groups()


class GradScaler:
    """Explicit single-optimizer loss scaling for ruda_torch.AdamW or Muon/MuonAdamW.

    Not torch.amp.GradScaler. Autocast is registered separately for RUDA dense projection ops. Accumulate scaled losses before one
    step/update pair. Finite checking happens in step. AdamW(fused_step=True,
    max_grad_norm=...) can clip the unscaled global L2 norm within its update,
    without modifying .grad. unscale_(), external in-place clipping, multiple
    optimizers per cycle and graph capture are not supported.
    """
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, min_scale=2.0**-24, max_scale=2.0**24):
        self._scale = _f32(init_scale, 'init_scale', positive=True)
        self.growth_factor = _f32(growth_factor, 'growth_factor', positive=True)
        self.backoff_factor = _f32(backoff_factor, 'backoff_factor', positive=True)
        self.min_scale = _f32(min_scale, 'min_scale', positive=True)
        self.max_scale = _f32(max_scale, 'max_scale', positive=True)
        if not (self.growth_factor > 1 and 0 < self.backoff_factor < 1 and self.min_scale <= self._scale <= self.max_scale):
            raise ValueError('invalid gradient scale bounds or factors')
        if type(growth_interval) is not int or growth_interval <= 0:
            raise ValueError('growth_interval must be a positive integer')
        self.growth_interval = growth_interval
        self._good = 0
        self._stage = 'ready'
        self._scaled = False
        self._overflow = False

    def scale(self, loss: torch.Tensor):
        if self._stage != 'ready':
            raise RuntimeError('call update() before scaling the next iteration')
        _tensor(loss, 'loss')
        if loss.numel() != 1 or loss.dtype != torch.float32:
            raise ValueError('use a one-element FP32 loss for explicit loss scaling')
        self._scaled = True
        return loss * self._scale

    def step(self, optimizer: AdamW, *args, **kwargs):
        supported = isinstance(optimizer, AdamW)
        if not supported and __package__:
            from .optim import Muon
            supported = isinstance(optimizer, Muon)
        if not supported:
            raise TypeError('this GradScaler supports ruda_torch.AdamW, Muon and MuonAdamW')
        if args or kwargs:
            raise TypeError('scaled optimizer step does not support closures or extra arguments')
        if self._stage != 'ready' or not self._scaled:
            raise RuntimeError('scale/backward must precede exactly one step per update')
        if not any(p.grad is not None and p.numel() for group in optimizer.param_groups for p in group['params']):
            raise RuntimeError('no gradients were produced')
        # A native failure leaves the scaler non-ready; reconstruct/reload after
        # recovery rather than accidentally unscaling the same gradients twice.
        self._stage = 'failed'
        result = optimizer.step(loss_scale=self._scale)
        self._overflow = optimizer.last_step_skipped
        self._stage = 'stepped'
        return result

    def update(self):
        if self._stage != 'stepped':
            raise RuntimeError('update() requires one completed optimizer step')
        if self._overflow:
            self._scale = max(self.min_scale, self._scale * self.backoff_factor)
            self._good = 0
        else:
            self._good += 1
            if self._good == self.growth_interval:
                self._scale = min(self.max_scale, self._scale * self.growth_factor)
                self._good = 0
        self._stage = 'ready'; self._scaled = False

    def get_scale(self):
        return self._scale

    def state_dict(self):
        if self._stage != 'ready' or self._scaled:
            raise RuntimeError('checkpoint scaler only between completed iterations')
        return {'version':1, 'scale':self._scale, 'good_steps':self._good,
                'growth_factor':self.growth_factor, 'backoff_factor':self.backoff_factor,
                'growth_interval':self.growth_interval, 'min_scale':self.min_scale, 'max_scale':self.max_scale}

    def load_state_dict(self, state):
        expected = {'version','scale','good_steps','growth_factor','backoff_factor','growth_interval','min_scale','max_scale'}
        if set(state) != expected or state['version'] != 1:
            raise ValueError('invalid RUDA GradScaler checkpoint')
        replacement = GradScaler(state['scale'], state['growth_factor'], state['backoff_factor'],
                                 state['growth_interval'], state['min_scale'], state['max_scale'])
        good = state['good_steps']
        if type(good) is not int or not 0 <= good < replacement.growth_interval:
            raise ValueError('invalid scaler growth counter')
        replacement._good = good
        self.__dict__.update(replacement.__dict__)
