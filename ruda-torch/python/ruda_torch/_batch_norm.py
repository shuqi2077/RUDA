import math
import torch
from torch._prims_common import suggest_memory_format
from . import _C
from ._ops import _dtypes, _registry, fill_, mean_dim, sum_dim, where


def _metadata(a, weight, bias, running_mean, running_var):
    if a.dtype not in _dtypes or a.ndim < 2 or a.shape[1] == 0:
        raise RuntimeError("batch normalization requires a supported floating tensor with a nonempty channel dimension")
    params = [v for v in (weight, bias, running_mean, running_var) if v is not None]
    dtype = params[0].dtype if params else a.dtype
    if dtype not in (a.dtype, torch.float32):
        raise RuntimeError("batch normalization parameters must match the input dtype or use float32")
    for value in params:
        if value.device != a.device or value.dtype != dtype or tuple(value.shape) != (a.shape[1],):
            raise RuntimeError("batch normalization parameters must share a device, channel shape and dtype")
    if (running_mean is None) != (running_var is None):
        raise RuntimeError("running_mean and running_var must either both be present or both be absent")
    axes = (0, *range(2, a.ndim))
    return axes, (1, a.shape[1], *((1,) * (a.ndim - 2))), math.prod(a.shape[d] for d in axes), dtype


def _invstd(variance, eps):
    result = torch.rsqrt(variance + eps)
    return where(variance == 0, 0.0, result) if eps == 0 else result


def _output(value, a):
    result = torch.empty(a.shape, device=a.device, dtype=a.dtype, memory_format=suggest_memory_format(a))
    return result.copy_(value)


def native_batch_norm(a, weight, bias, running_mean, running_var, training, momentum, eps):
    axes, broadcast, count, _ = _metadata(a, weight, bias, running_mean, running_var)
    if not training and running_mean is None:
        raise RuntimeError("batch normalization inference requires running statistics")
    if count == 0:
        stats = lambda: torch.empty((a.shape[1],), device=a.device, dtype=torch.float32)
        return _output(a, a), stats(), stats()
    x = a.float()
    if training:
        if running_mean is not None:
            _C.check_index_output(running_mean, a, running_var)
            _C.check_index_output(running_var, a, running_mean)
        mean = mean_dim(x, axes)
        centered = x - mean.view(broadcast)
        variance = mean_dim(centered * centered, axes)
        invstd = _invstd(variance, eps)
        if running_mean is not None:
            updated_mean = running_mean.float() * (1 - momentum) + mean * momentum
            unbiased = variance * count / (count - 1)
            updated_var = running_var.float() * (1 - momentum) + unbiased * momentum
            running_mean.copy_(updated_mean)
            running_var.copy_(updated_var)
    else:
        mean = running_mean.float().clone()
        invstd = _invstd(running_var.float(), eps)
        centered = x - mean.view(broadcast)
    result = centered * invstd.view(broadcast)
    if weight is not None:
        result = result * weight.float().view(broadcast)
    if bias is not None:
        result = result + bias.float().view(broadcast)
    return _output(result, a), mean, invstd


def native_batch_norm_backward(grad, a, weight, running_mean, running_var, save_mean, save_invstd,
                               train, eps, output_mask):
    axes, broadcast, count, param_dtype = _metadata(a, weight, None, running_mean, running_var)
    if grad.shape != a.shape or grad.device != a.device or grad.dtype != a.dtype or len(output_mask) != 3:
        raise RuntimeError("invalid batch normalization gradient or output mask")
    if count == 0:
        zero = lambda: fill_(torch.empty((a.shape[1],), device=a.device, dtype=param_dtype), 0)
        return (_output(grad, a) if output_mask[0] else None,
                zero() if output_mask[1] else None, zero() if output_mask[2] else None)
    saved = save_mean is not None and save_invstd is not None and save_mean.numel() != 0 and save_invstd.numel() != 0
    if saved:
        for value in (save_mean, save_invstd):
            if value.device != a.device or value.dtype != torch.float32 or tuple(value.shape) != (a.shape[1],):
                raise RuntimeError("invalid saved batch normalization statistics")
        mean, invstd = save_mean, save_invstd
    elif train:
        raise RuntimeError("batch normalization training backward requires saved statistics")
    elif running_mean is not None:
        mean, invstd = running_mean.float(), _invstd(running_var.float(), eps)
    else:
        raise RuntimeError("batch normalization inference backward requires statistics")
    dy = grad.float()
    centered = a.float() - mean.view(broadcast)
    total = sum_dim(dy, axes)
    dot = sum_dim(dy * centered, axes)
    dx = dw = db = None
    if output_mask[0]:
        value = dy
        if train:
            value = value - (total / count).view(broadcast)
            value = value - centered * (dot * invstd * invstd / count).view(broadcast)
        value = value * invstd.view(broadcast)
        if weight is not None:
            value = value * weight.float().view(broadcast)
        dx = _output(value, a)
    if output_mask[1]:
        dw = (dot * invstd).to(param_dtype)
    if output_mask[2]:
        db = total.to(param_dtype)
    return dx, dw, db


def _no_stats(a, weight, bias, training, momentum, eps):
    return native_batch_norm(a, weight, bias, None, None, training, momentum, eps)


def _no_training(a, weight, bias, running_mean, running_var, momentum, eps):
    return native_batch_norm(a, weight, bias, running_mean, running_var, False, momentum, eps)


for name, function in {
    "native_batch_norm": native_batch_norm,
    "native_batch_norm_backward": native_batch_norm_backward,
    "_native_batch_norm_legit": native_batch_norm,
    "_native_batch_norm_legit.no_stats": _no_stats,
    "_native_batch_norm_legit_no_training": _no_training,
}.items():
    _registry.impl(name, function)
