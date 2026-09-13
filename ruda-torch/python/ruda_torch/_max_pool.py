import torch
from torch._prims_common import suggest_memory_format
from . import _C
from ._ops import _registry
from ._spatial import _matching, _parameters


def _shape(a, kernel_size, stride, padding, dilation, ceil_mode):
    _matching(a)
    if a.ndim not in (3, 4) or any(s <= 0 for s in a.shape[-3:]):
        raise RuntimeError("max_pool2d requires a 3D or 4D tensor with nonempty channels and spatial dimensions")
    kernel = _parameters(kernel_size, 2, "max pooling kernel", minimum=1)
    stride = _parameters(stride, 2, "max pooling stride", default=kernel, minimum=1)
    padding = _parameters(padding, 2, "max pooling padding")
    dilation = _parameters(dilation, 2, "max pooling dilation", minimum=1)
    if any(p > k // 2 for p, k in zip(padding, kernel)):
        raise RuntimeError("max pooling padding must not exceed half the kernel size")
    output = []
    for size, k, s, p, d in zip(a.shape[-2:], kernel, stride, padding, dilation):
        extent = d * (k - 1) + 1
        length = (size + 2 * p - extent + (s - 1 if ceil_mode else 0)) // s + 1
        if ceil_mode and (length - 1) * s >= size + p:
            length -= 1
        if length <= 0:
            raise RuntimeError("max pooling output spatial dimensions must be positive")
        if size + p > 2147483647 or extent > 2147483647 or (length - 1) * s + extent > 4294967295:
            raise NotImplementedError("RUDA max pooling spatial coordinates exceed the kernel's integer range")
        output.append(length)
    channels_last = a.ndim == 4 and suggest_memory_format(a) == torch.channels_last
    return (*a.shape[:-2], *output), (*kernel, *stride, *padding, *dilation, int(ceil_mode), int(channels_last))


def max_pool2d_with_indices(a, kernel_size, stride=(), padding=(0,), dilation=(1,), ceil_mode=False):
    shape, params = _shape(a, kernel_size, stride, padding, dilation, ceil_mode)
    x = a if a.ndim == 4 else a.unsqueeze(0)
    batched_shape = shape if a.ndim == 4 else (1, *shape)
    memory_format = suggest_memory_format(a) if a.ndim == 4 else torch.contiguous_format
    values = torch.empty(batched_shape, device=a.device, dtype=a.dtype, memory_format=memory_format)
    indices = torch.empty(batched_shape, device=a.device, dtype=torch.int64, memory_format=memory_format)
    if values.numel():
        _C.spatial(6, x, indices, values, params)
    if a.ndim == 3:
        values, indices = values.squeeze(0), indices.squeeze(0)
    return values, indices


def max_pool2d_with_indices_backward(grad, a, kernel_size, stride, padding, dilation, ceil_mode, indices):
    shape, params = _shape(a, kernel_size, stride, padding, dilation, ceil_mode)
    _matching(a, grad)
    if tuple(grad.shape) != shape or tuple(indices.shape) != shape:
        raise RuntimeError("max pooling gradients and indices must match the forward output shape")
    if indices.dtype != torch.int64 or indices.device != a.device:
        raise RuntimeError("max pooling indices must use int64 on the input device")
    dy = grad.float()
    batched_shape = a.shape
    if a.ndim == 3:
        dy, indices = dy.unsqueeze(0), indices.unsqueeze(0)
        batched_shape = (1, *a.shape)
    result = torch.empty(batched_shape, device=a.device, dtype=torch.float32,
                         memory_format=suggest_memory_format(a) if a.ndim == 4 else torch.contiguous_format)
    if result.numel():
        _C.spatial(7, dy, indices, result, params)
    return (result if a.ndim == 4 else result.squeeze(0)).to(a.dtype)


for name, function in {
    "max_pool2d_with_indices": max_pool2d_with_indices,
    "max_pool2d_with_indices_backward": max_pool2d_with_indices_backward,
}.items():
    _registry.impl(name, function)
