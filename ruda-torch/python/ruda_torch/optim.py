"""PyTorch training integration for RUDA's Muon conventions.

This is a torch.optim.Optimizer implemented with same-device tensor operations.
On RUDA, those operations dispatch to the existing Rust-backed tensor kernels.
It does NOT wrap a ruda_optim::Muon Rust object or introduce a fused Muon ABI.
Momentum, normalization, matrix layout, LR adjustment and decay conventions
are explicit and match ruda-optim/src/optim/muon/mod.rs (not byte-identical
checkpoint formats). No torch.distributed collective is hidden in step().
"""
from __future__ import annotations

import copy
import math
import torch
from torch import nn

from ._architecture_ops import maximum_abs, work_dtype, precision_context, positive_float


def _number(value, name, *, nonnegative=False):
    if isinstance(value, (bool, torch.Tensor)):
        raise TypeError(f"{name} must be a Python number")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def _finite(x):
    return ((x == x) & (x.abs() <= torch.finfo(x.dtype).max)).all()


def _lower_bound(x, minimum):
    return torch.where(x > minimum, x, torch.full_like(x, minimum))


def muon_orthogonalize(gradient: torch.Tensor, *, ns_steps=5,
                       coefficients=(3.4445, -4.775, 2.0315), eps=1e-7,
                       stable_normalization=True) -> torch.Tensor:
    """Newton-Schulz quintic iteration on one complete 2-D gradient matrix.

    Uses FP32 work for half/BF16 and FP64 for double. Frobenius normalization
    uses max(norm, eps), as in RUDA Rust; it is not the upstream BF16 norm+eps
    variant. This finite iteration is approximate, NOT an exact polar factor.
    """
    if gradient.ndim != 2 or min(gradient.shape) < 1:
        raise ValueError("Muon requires a nonempty 2-D matrix")
    if type(ns_steps) is not int or not 1 <= ns_steps < 100:
        raise ValueError("ns_steps must be an integer in [1, 99]")
    if len(coefficients) != 3:
        raise ValueError("coefficients must contain three finite numbers")
    a, b, c = (_number(v, "coefficient") for v in coefficients)
    eps = positive_float(eps, "eps")
    with precision_context(gradient):
        x = gradient.to(work_dtype(gradient))
        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.t()
        if stable_normalization:
            scale = _lower_bound(maximum_abs(x), torch.finfo(x.dtype).tiny)
            x = x / scale
            norm = x.square().sum().sqrt()
            floor = eps / scale
            norm = torch.where(norm > floor, norm, floor)
        else:
            norm = _lower_bound(x.square().sum().sqrt(), eps)
        x = x / norm
        for _ in range(ns_steps):
            gram = x @ x.t()
            polynomial = b * gram + c * (gram @ gram)
            x = a * x + polynomial @ x
        return x.t() if transposed else x


def _validate_group(group):
    for name in ("lr", "weight_decay"):
        _number(group[name], name, nonnegative=True)
    if not math.isfinite(group['lr'] * group['weight_decay']):
        raise ValueError("lr * weight_decay must be finite")
    if type(group['use_muon']) is not bool:
        raise TypeError("use_muon must be a bool")
    positive_float(group['eps'], 'eps')
    if group['use_muon']:
        beta = _number(group['momentum'], 'momentum')
        dampening = _number(group['dampening'], 'dampening')
        if not 0 <= beta < 1 or not 0 <= dampening <= 1:
            raise ValueError('invalid momentum/dampening')
        if type(group['nesterov']) is not bool or type(group['flatten']) is not bool:
            raise TypeError('nesterov and flatten must be bool')
        if group['nesterov'] and (beta == 0 or dampening != 0):
            raise ValueError('Nesterov requires positive momentum and zero dampening')
        if group['momentum_mode'] not in ('sgd', 'ema'):
            raise ValueError('momentum_mode must be sgd or ema')
        if group['momentum_mode'] == 'ema' and dampening != 0:
            raise ValueError('EMA requires zero dampening')
        if group['adjust_lr'] not in ('original', 'match_rms_adamw'):
            raise ValueError('adjust_lr must be original or match_rms_adamw')
        if group['matrix_layout'] not in ('as_stored', 'input_output'):
            raise ValueError('matrix_layout must be as_stored or input_output')
        if type(group['ns_steps']) is not int or not 1 <= group['ns_steps'] < 100:
            raise ValueError('ns_steps must be in [1, 99]')
        if len(group['ns_coefficients']) != 3:
            raise ValueError('ns_coefficients must contain three numbers')
        for c in group['ns_coefficients']:
            _number(c, 'ns coefficient')
    else:
        if len(group['betas']) != 2 or any(not 0 <= _number(v, 'beta') < 1 for v in group['betas']):
            raise ValueError('AdamW betas must be in [0, 1)')


def _parameter(p, group):
    if not isinstance(p, torch.Tensor) or not p.is_floating_point() or p.layout != torch.strided:
        raise TypeError('parameters must be dense floating-point tensors')
    work_dtype(p)
    if not p.is_contiguous() or p.is_conj() or p.is_neg() or p.numel() == 0:
        raise ValueError('parameters must be nonempty contiguous tensors without unresolved view bits')
    if group['use_muon'] and (p.ndim < 2 or (p.ndim > 2 and not group['flatten'])):
        raise ValueError('Muon parameters must be matrices; use flatten=True explicitly for convolution weights')


def _overlap_check(parameters, gradients=()):
    # Storage metadata only; O(n log n), not O(n^2) in model parameter count.
    # Read-only gradient views may overlap each other, but never parameters.
    storages = {}
    for kind, tensors in (('parameter', parameters), ('gradient', gradients)):
        for tensor in tensors:
            if tensor.layout != torch.strided or not tensor.is_contiguous():
                raise ValueError('optimizer parameters/gradients must be dense and contiguous')
            if not tensor.numel():
                continue
            base = (tensor.device, tensor.untyped_storage()._cdata)
            start = tensor.storage_offset() * tensor.element_size()
            storages.setdefault(base, []).append((start, start + tensor.numel() * tensor.element_size(), kind))
    for intervals in storages.values():
        previous_parameter_end = previous_any_end = -1
        for start, end, kind in sorted(intervals):
            if (kind == 'parameter' and start < previous_any_end) or start < previous_parameter_end:
                raise ValueError('optimizer parameters must not overlap parameters or gradients')
            if kind == 'parameter':
                previous_parameter_end = max(previous_parameter_end, end)
            previous_any_end = max(previous_any_end, end)


class Muon(torch.optim.Optimizer):
    """Single-device eager Muon with optional explicit AdamW parameter groups.

    FP32 master weights/moments for FP16/BF16; gradients remain read-only.
    NaN/Inf in any active gradient skips the WHOLE step, without decay/counter
    updates. Two scalar finite-status readbacks guard preflight and tentative
    updates. This is intentionally not graph-capturable. Temporary proposed
    states/weights cost additional memory, and driver failures during commit
    cannot be made transactional. Distributed gradient synchronization is the
    caller's responsibility; orthogonalize complete matrices, not local shards.
    """
    def __init__(self, params, lr=0.02, *, momentum=0.95, weight_decay=0.0,
                 nesterov=True, momentum_mode='sgd', dampening=0.0,
                 ns_steps=5, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-7,
                 adjust_lr='original', matrix_layout='as_stored',
                 stable_normalization=True, flatten=False, max_grad_norm=None):
        self.max_grad_norm = None if max_grad_norm is None else positive_float(max_grad_norm, 'max_grad_norm')
        self.last_step_skipped = False
        self._expected_versions = {}
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
            momentum_mode=momentum_mode, dampening=dampening, ns_steps=ns_steps,
            ns_coefficients=tuple(ns_coefficients), eps=eps, adjust_lr=adjust_lr,
            matrix_layout=matrix_layout, stable_normalization=bool(stable_normalization),
            flatten=flatten, use_muon=True, betas=(0.9, 0.999))
        super().__init__(params, defaults)
        self._validate_all()

    def add_param_group(self, param_group):
        # Check configuration before mutating Optimizer's group list.
        group = dict(self.defaults, **param_group)
        group['params'] = [group['params']] if isinstance(group['params'], torch.Tensor) else list(group['params'])
        _validate_group(group)
        for p in group['params']:
            _parameter(p, group)
        previous = [p for g in self.param_groups for p in g['params']]
        _overlap_check(previous + group['params'])
        devices = {p.device for p in previous + group['params']}
        if len(devices) > 1:
            raise ValueError('Muon supports a single device per optimizer')
        return super().add_param_group(group)

    def _validate_all(self):
        parameters = []
        for group in self.param_groups:
            _validate_group(group)
            for p in group['params']:
                _parameter(p, group)
                parameters.append(p)
        _overlap_check(parameters)
        if len({p.device for p in parameters}) > 1:
            raise ValueError('Muon supports a single device per optimizer')

    def _state_check(self, p, group, state):
        if not state:
            return
        expected = 'muon:' + group['momentum_mode'] if group['use_muon'] else 'adamw'
        if state.get('algorithm') != expected:
            raise ValueError('optimizer algorithm/momentum convention changed; reset or restore matching state')
        if type(state.get('step')) is not int or state['step'] < 0:
            raise ValueError('invalid optimizer step counter')
        names = ('master', 'momentum_buffer') if group['use_muon'] else ('master', 'exp_avg', 'exp_avg_sq')
        for name in names:
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != p.shape or value.dtype != work_dtype(p) or value.device != p.device:
                raise ValueError(f'invalid optimizer {name} shape/device/dtype')

    @torch.no_grad()
    def step(self, closure=None, *, loss_scale=1.0):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        scale = positive_float(loss_scale, 'loss_scale')
        self._validate_all()
        parameters = [p for group in self.param_groups for p in group['params']]
        active = [(p, group) for group in self.param_groups for p in group['params'] if p.grad is not None]
        if not active:
            self.last_step_skipped = False
            return loss
        _overlap_check(parameters, [p.grad for p, _ in active])
        flags, prepared = [], []
        for p, group in active:
            grad = p.grad
            if grad.is_sparse or grad.shape != p.shape or grad.device != p.device or grad.dtype != p.dtype:
                raise ValueError('gradients must be dense, same shape/device/dtype as their parameters')
            state = self.state.get(p, {})
            self._state_check(p, group, state)
            if state and id(p) in self._expected_versions and p._version != self._expected_versions[id(p)]:
                raise ValueError('parameter changed outside the optimizer; reload/reset master state')
            g = grad.to(work_dtype(p)) / scale
            flags.extend((_finite(g), _finite(p)))
            if state:
                flags.extend(_finite(v) for v in state.values() if isinstance(v, torch.Tensor))
            prepared.append((p, group, state, g))
        if not bool(torch.stack(flags).all().item()):
            self.last_step_skipped = True
            return loss
        clip = None
        if self.max_grad_norm is not None:
            scales = torch.stack([maximum_abs(g) for _, _, _, g in prepared])
            largest = _lower_bound(maximum_abs(scales), torch.finfo(scales.dtype).tiny)
            normalized_norm = torch.stack([(g / largest).square().sum() for _, _, _, g in prepared]).sum().sqrt()
            raw = (self.max_grad_norm / largest) / _lower_bound(normalized_norm, torch.finfo(scales.dtype).tiny)
            clip = torch.where(raw < 1, raw, torch.ones_like(raw))
        proposals, flags = [], []
        for p, group, old, g in prepared:
            with precision_context(p):
                if clip is not None:
                    g = g * clip.to(g.dtype)
                master = old['master'] if old else p.to(work_dtype(p)).clone()
                step = old.get('step', 0) + 1
                if group['use_muon']:
                    beta = group['momentum']
                    if group['momentum_mode'] == 'ema':
                        buffer = (old['momentum_buffer'] * beta if old else torch.zeros_like(g)) + g * (1-beta)
                        update = g * (1-beta) + buffer * beta if group['nesterov'] else buffer
                    else:
                        buffer = old['momentum_buffer'] * beta + g * (1-group['dampening']) if old else g.clone()
                        update = g + buffer * beta if group['nesterov'] else buffer
                    matrix = update.reshape(p.shape[0], -1)
                    update = muon_orthogonalize(matrix, ns_steps=group['ns_steps'],
                        coefficients=group['ns_coefficients'], eps=group['eps'],
                        stable_normalization=group['stable_normalization']).reshape(p.shape)
                    rows, cols = matrix.shape
                    if group['matrix_layout'] == 'input_output':
                        rows, cols = cols, rows
                    ratio = math.sqrt(max(1, rows/cols)) if group['adjust_lr'] == 'original' else 0.2 * math.sqrt(max(rows, cols))
                    value = master * (1-group['lr']*group['weight_decay']) - update * (group['lr']*ratio)
                    new = dict(step=step, algorithm='muon:'+group['momentum_mode'], momentum_buffer=buffer, master=value)
                else:
                    beta1, beta2 = group['betas']
                    first = (old['exp_avg'] * beta1 if old else torch.zeros_like(g)) + g * (1-beta1)
                    second = (old['exp_avg_sq'] * beta2 if old else torch.zeros_like(g)) + g.square() * (1-beta2)
                    update = (first/(1-beta1**step)) / ((second/(1-beta2**step)).sqrt() + group['eps'])
                    value = master * (1-group['lr']*group['weight_decay']) - group['lr'] * update
                    new = dict(step=step, algorithm='adamw', exp_avg=first, exp_avg_sq=second, master=value)
                stored = value.to(p.dtype)
                flags.extend(_finite(v) for v in new.values() if isinstance(v, torch.Tensor))
                flags.append(_finite(stored))
                proposals.append((p, new, stored))
        if not bool(torch.stack(flags).all().item()):
            self.last_step_skipped = True
            return loss
        for p, state, value in proposals:
            p.copy_(value)
            self.state[p] = state
            self._expected_versions[id(p)] = p._version
        self.last_step_skipped = False
        return loss

    def state_dict(self):
        result = super().state_dict()
        result['ruda_muon'] = {'version': 1, 'max_grad_norm': self.max_grad_norm}
        return result

    def load_state_dict(self, state_dict):
        metadata = state_dict.get('ruda_muon')
        if not isinstance(metadata, dict) or metadata.get('version') != 1:
            raise ValueError('expected a version-1 ruda_torch Muon checkpoint, not a Rust/upstream state')
        norm = metadata.get('max_grad_norm')
        if norm is not None:
            positive_float(norm, 'max_grad_norm')
        saved_groups = state_dict['param_groups']
        if len(saved_groups) != len(self.param_groups) or any(len(a['params']) != len(b['params']) for a, b in zip(saved_groups, self.param_groups)):
            raise ValueError('optimizer parameter groups do not match the checkpoint')
        restored = {}
        for current, saved in zip(self.param_groups, saved_groups):
            _validate_group(saved)
            for p, key in zip(current['params'], saved['params']):
                _parameter(p, saved)
                original = state_dict['state'].get(key, {})
                # Read original FP32 tensors BEFORE base Optimizer loads/casts
                # them to a half parameter dtype, which would irreversibly round.
                value = {name: tensor.detach().to(device=p.device, dtype=work_dtype(p)).clone()
                         if isinstance(tensor, torch.Tensor) else copy.deepcopy(tensor)
                         for name, tensor in original.items()}
                self._state_check(p, saved, value)
                if value:
                    restored[p] = value
        super().load_state_dict({k: v for k, v in state_dict.items() if k != 'ruda_muon'})
        for p, value in restored.items():
            self.state[p] = value
        self.max_grad_norm = norm
        self._expected_versions = {id(p): p._version for p in restored}
        self.last_step_skipped = False


class MuonAdamW(Muon):
    """One optimizer/checkpoint with explicit Muon and AdamW parameter groups."""
    def __init__(self, param_groups, **kwargs):
        groups = list(param_groups)
        if not groups or any(not isinstance(group, dict) or 'use_muon' not in group for group in groups):
            raise ValueError('MuonAdamW requires explicit groups with use_muon=True/False')
        super().__init__(groups, **kwargs)

    @classmethod
    def from_model(cls, model: nn.Module, *, muon_modules, adamw_modules=(),
                   lr=0.02, adamw_lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), **kwargs):
        """Nominate hidden modules explicitly; embeddings and exclusions use AdamW.

        Parameter identity, not name/shape alone, determines membership. Shared
        embedding/head weights are included once, with AdamW taking precedence.
        No generic model wrapper can infer which arbitrary matrix is a head.
        """
        all_parameters = list(model.parameters())
        known = {id(p) for p in all_parameters}
        candidates = {id(p) for m in muon_modules for p in m.parameters()}
        excluded = {id(p) for m in adamw_modules for p in m.parameters()}
        excluded |= {id(p) for m in model.modules() if isinstance(m, (nn.Embedding, nn.EmbeddingBag)) for p in m.parameters()}
        if (candidates | excluded) - known:
            raise ValueError('parameter-group modules must belong to model')
        matrices, others = [], []
        flatten = kwargs.get('flatten', False)
        for p in all_parameters:
            if not p.requires_grad:
                continue
            eligible = id(p) in candidates and id(p) not in excluded and (p.ndim == 2 or (p.ndim > 2 and flatten))
            (matrices if eligible else others).append(p)
        groups = []
        if matrices:
            groups.append(dict(params=matrices, use_muon=True, lr=lr, weight_decay=weight_decay))
        if others:
            groups.append(dict(params=others, use_muon=False, lr=adamw_lr, weight_decay=weight_decay, betas=betas, eps=1e-8))
        if not matrices:
            raise ValueError('no eligible hidden matrices were nominated for Muon')
        return cls(groups, **kwargs)
