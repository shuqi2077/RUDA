import torch
from torch._prims_common import suggest_memory_format
from . import _C
from ._ops import _dtypes, _registry, fill_, sum_dim


def _parameters(values, rank, name, *, default=None, minimum=0):
    values = tuple(values)
    if not values and default is not None:
        values = default
    if len(values) == 1:
        values = values * rank
    if len(values) != rank or any(v < minimum or v > 2147483647 for v in values):
        raise RuntimeError(f"invalid {name} for {rank} spatial dimensions")
    return values


def _matching(a, *others):
    if a.dtype not in _dtypes:
        raise RuntimeError("RUDA spatial operators require float32, float16 or bfloat16")
    for other in others:
        if other is not None and (other.device != a.device or other.dtype != a.dtype):
            raise RuntimeError("RUDA spatial tensors must have matching devices and dtypes")


def _empty(shape, a):
    return torch.empty(shape, device=a.device, dtype=a.dtype, memory_format=suggest_memory_format(a))


def _conv_shape(a, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    _matching(a, weight, bias)
    rank = a.ndim - 2
    if rank not in (1, 2, 3) or weight.ndim != a.ndim:
        raise RuntimeError("convolution requires batched 1D, 2D or 3D input and matching weight rank")
    stride = _parameters(stride, rank, "convolution stride", minimum=1)
    padding = _parameters(padding, rank, "convolution padding")
    dilation = _parameters(dilation, rank, "convolution dilation", minimum=1)
    output_padding = _parameters(output_padding, rank, "convolution output_padding")
    if groups <= 0 or any(s <= 0 for s in a.shape[1:]) or any(s <= 0 for s in weight.shape):
        raise RuntimeError("convolution groups, channels, spatial sizes and kernel sizes must be positive")
    if transposed:
        if a.shape[1] != weight.shape[0] or weight.shape[0] % groups:
            raise RuntimeError("transposed convolution channels do not match grouped weights")
        if any(o >= s and o >= d for o, s, d in zip(output_padding, stride, dilation)):
            raise RuntimeError("output_padding must be smaller than stride or dilation")
        channels = weight.shape[1] * groups
        spatial = tuple((v - 1) * s - 2 * p + d * (k - 1) + o + 1
                        for v, k, s, p, d, o in zip(a.shape[2:], weight.shape[2:], stride, padding, dilation, output_padding))
    else:
        if any(output_padding):
            raise RuntimeError("output_padding is only supported for transposed convolution")
        if a.shape[1] != weight.shape[1] * groups or weight.shape[0] % groups:
            raise RuntimeError("convolution channels do not match grouped weights")
        channels = weight.shape[0]
        spatial = tuple((v + 2 * p - d * (k - 1) - 1) // s + 1
                        for v, k, s, p, d in zip(a.shape[2:], weight.shape[2:], stride, padding, dilation))
    if any(v <= 0 for v in spatial):
        raise RuntimeError("convolution output spatial dimensions must be positive")
    if bias is not None and tuple(bias.shape) != (channels,):
        raise RuntimeError("convolution bias must have one value per output channel")
    return (a.shape[0], channels, *spatial), (*stride, *padding, *dilation, *output_padding, groups)


def _run_conv(op, a, b, shape, params):
    result = _empty(shape, a)
    if result.numel():
        _C.spatial(op, a, b, result, params)
    return result


def convolution(a, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    shape, params = _conv_shape(a, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
    result = _run_conv(3 if transposed else 0, a.float(), weight.float(), shape, params)
    if bias is not None:
        result = result + bias.float().view(1, -1, *((1,) * (a.ndim - 2)))
    return result.to(a.dtype).contiguous(memory_format=suggest_memory_format(a))


def _convolution(a, weight, bias, stride, padding, dilation, transposed, output_padding, groups,
                 benchmark, deterministic, cudnn_enabled, allow_tf32=True):
    return convolution(a, weight, bias, stride, padding, dilation, transposed, output_padding, groups)


def convolution_backward(grad, a, weight, bias_sizes, stride, padding, dilation, transposed,
                         output_padding, groups, output_mask):
    shape, params = _conv_shape(a, weight, None, stride, padding, dilation, transposed, output_padding, groups)
    _matching(a, grad)
    if tuple(grad.shape) != shape or len(output_mask) != 3:
        raise RuntimeError("invalid convolution gradient shape or output mask")
    if bias_sizes is not None and tuple(bias_sizes) != (shape[1],):
        raise RuntimeError("invalid convolution bias gradient shape")
    x, w, dy = a.float(), weight.float(), grad.float()
    dx = dw = db = None
    if a.shape[0] == 0:
        if output_mask[0]:
            dx = _empty(a.shape, a)
        if output_mask[1]:
            dw = fill_(_empty(weight.shape, weight), 0)
        if output_mask[2]:
            db = fill_(torch.empty((shape[1],), device=a.device, dtype=a.dtype), 0)
        return dx, dw, db
    if output_mask[0]:
        if transposed:
            rank = a.ndim - 2
            forward_params = (*params[:3 * rank], *((0,) * rank), groups)
            dx_shape, _ = _conv_shape(dy, w, None, stride, padding, dilation, False, (0,), groups)
            value = _run_conv(0, dy, w, dx_shape, forward_params)
            for axis in range(2, a.ndim):
                value = value.narrow(axis, 0, a.shape[axis])
        else:
            value = _run_conv(1, dy, w, a.shape, params)
        dx = value.to(a.dtype).contiguous(memory_format=suggest_memory_format(a))
    if output_mask[1]:
        value = _run_conv(2, dy if transposed else x, x if transposed else dy, weight.shape, params)
        dw = value.to(weight.dtype).contiguous(memory_format=suggest_memory_format(weight))
    if output_mask[2]:
        db = sum_dim(dy, (0, *range(2, dy.ndim))).to(a.dtype)
    return dx, dw, db


def _pool_shape(a, kernel_size, stride, padding, ceil_mode, divisor_override):
    _matching(a)
    if a.ndim not in (3, 4) or any(s <= 0 for s in a.shape[-3:]):
        raise RuntimeError("avg_pool2d requires nonempty channels and spatial dimensions in a 3D or 4D input")
    kernel = _parameters(kernel_size, 2, "pooling kernel", minimum=1)
    stride = _parameters(stride, 2, "pooling stride", default=kernel, minimum=1)
    padding = _parameters(padding, 2, "pooling padding")
    if any(p > k // 2 for p, k in zip(padding, kernel)):
        raise RuntimeError("pooling padding must not exceed half the kernel size")
    if divisor_override == 0:
        raise RuntimeError("average pooling divisor must be nonzero")
    spatial = []
    for size, k, s, p in zip(a.shape[-2:], kernel, stride, padding):
        out = (size + 2 * p - k + (s - 1 if ceil_mode else 0)) // s + 1
        if ceil_mode and (out - 1) * s >= size + p:
            out -= 1
        if out <= 0:
            raise RuntimeError("pooling output spatial dimensions must be positive")
        spatial.append(out)
    return (*a.shape[:-2], *spatial), (*kernel, *stride, *padding)


def avg_pool2d(a, kernel_size, stride=(), padding=(0,), ceil_mode=False, count_include_pad=True, divisor_override=None):
    shape, params = _pool_shape(a, kernel_size, stride, padding, ceil_mode, divisor_override)
    x = a.float()
    if a.ndim == 3:
        x = x.unsqueeze(0)
    result = _empty(shape if a.ndim == 4 else (1, *shape), x)
    if result.numel():
        _C.spatial(4, x, x, result, (*params, int(count_include_pad), int(ceil_mode), divisor_override or 0))
    return (result if a.ndim == 4 else result.squeeze(0)).to(a.dtype)


def avg_pool2d_backward(grad, a, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    shape, params = _pool_shape(a, kernel_size, stride, padding, ceil_mode, divisor_override)
    _matching(a, grad)
    if tuple(grad.shape) != shape:
        raise RuntimeError("average pooling gradient shape does not match the forward output")
    x, dy = a.float(), grad.float()
    if a.ndim == 3:
        x, dy = x.unsqueeze(0), dy.unsqueeze(0)
    result = _empty(x.shape, x)
    if result.numel():
        _C.spatial(5, x, dy, result, (*params, int(count_include_pad), int(ceil_mode), divisor_override or 0))
    return (result if a.ndim == 4 else result.squeeze(0)).to(a.dtype)


for name, function in {
    "convolution": convolution,
    "_convolution": _convolution,
    "_convolution.deprecated": _convolution,
    "convolution_backward": convolution_backward,
    "avg_pool2d": avg_pool2d,
    "avg_pool2d_backward": avg_pool2d_backward,
}.items():
    _registry.impl(name, function)
