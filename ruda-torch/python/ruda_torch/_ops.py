import math
import torch
from . import _C

_registry = torch.library.Library("aten", "IMPL", "PrivateUse1")
_dtypes = (torch.float32, torch.float16, torch.bfloat16)

def _out(shape, tensor):
    return torch.empty(shape, device=tensor.device, dtype=tensor.dtype)

def _apply(op, a, b=None, *, out=None, scalar=0.0):
    if b is None:
        b = a
    a, b = torch.broadcast_tensors(a, b)
    if out is None:
        out = _out(a.shape, a)
    if out.shape != a.shape:
        raise RuntimeError("RUDA output shape mismatch")
    _C.execute(op, a, b, out, scalar)
    return out

def _scalar(value, tensor):
    if isinstance(value, torch.Tensor):
        if value.device != tensor.device or value.dtype != tensor.dtype:
            raise NotImplementedError("RUDA requires matching GPU devices and tensor dtypes")
        return value
    result = torch.empty((), device=tensor.device, dtype=torch.float32)
    return fill_(result, value)

def fill_(tensor, value):
    _C.execute(5, tensor, tensor, tensor, float(value))
    return tensor

def add(a, b, *, alpha=1):
    return _apply(1, a, _scalar(b, a), scalar=float(alpha))

def add_(a, b, *, alpha=1):
    return _apply(1, a, _scalar(b, a), out=a, scalar=float(alpha))

def mul(a, b):
    return _apply(2, a, _scalar(b, a))

def mul_(a, b):
    return _apply(2, a, _scalar(b, a), out=a)

def div(a, b, *, rounding_mode=None):
    if rounding_mode is not None:
        raise NotImplementedError("RUDA rounded division is not implemented")
    return _apply(8, a, _scalar(b, a))

def div_(a, b, *, rounding_mode=None):
    if rounding_mode is not None:
        raise NotImplementedError("RUDA rounded division is not implemented")
    return _apply(8, a, _scalar(b, a), out=a)

def mm(a, b):
    _scalar(b, a)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise RuntimeError("RUDA mm requires compatible two-dimensional tensors")
    out = _out((a.shape[0], b.shape[1]), a)
    _C.execute(7, a, b, out, 0.0)
    return out

def addmm(bias, a, b, *, beta=1, alpha=1):
    _scalar(b, a)
    _scalar(bias, a)
    if a.dtype != torch.float32:
        return addmm(bias.float(), a.float(), b.float(), beta=beta, alpha=alpha).to(a.dtype)
    result = mm(a, b)
    if alpha != 1:
        mul_(result, alpha)
    if beta != 0:
        add_(result, bias, alpha=beta)
    return result

def bmm(a, b):
    _scalar(b, a)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise RuntimeError("RUDA bmm requires compatible three-dimensional batches")
    out = _out((a.shape[0], a.shape[1], b.shape[2]), a)
    _C.execute(30, a, b, out, 0.0)
    return out

def _dimension(dim, rank):
    if dim < -max(rank, 1) or dim >= max(rank, 1):
        raise IndexError("dimension out of range")
    return dim % rank if rank else 0

def softmax(a, dim, half_to_float=False, *, logarithmic=False):
    if half_to_float:
        if a.dtype != torch.float16:
            raise RuntimeError("half_to_float requires float16 input")
        a = a.float()
    dim = _dimension(dim, a.ndim)
    out = _out(a.shape, a)
    _C.execute(33 if logarithmic else 31, a, a, out, float(dim))
    return out

def softmax_backward(grad, output, dim, input_dtype, *, logarithmic=False):
    if input_dtype not in _dtypes or grad.shape != output.shape:
        raise RuntimeError("RUDA softmax backward requires supported dtypes and matching shapes")
    dim = _dimension(dim, output.ndim)
    out = torch.empty(output.shape, device=output.device, dtype=input_dtype)
    _C.execute(34 if logarithmic else 32, grad, output, out, float(dim))
    return out

def sum_dim(a, dim=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        if dtype not in _dtypes:
            raise NotImplementedError("RUDA sum supports float32, float16 and bfloat16")
        a = a.to(dtype)
    rank = a.ndim
    if dim is None or len(dim) == 0:
        dimensions = tuple(range(rank))
    else:
        if any(d < -max(rank, 1) or d >= max(rank, 1) for d in dim):
            raise IndexError("sum dimension out of range")
        dimensions = tuple(d % rank if rank else 0 for d in dim)
        if len(set(dimensions)) != len(dimensions):
            raise RuntimeError("duplicate sum dimension")
    shape = tuple(1 if i in dimensions else d for i, d in enumerate(a.shape))
    out = _out(shape, a)
    _C.execute(6, a, a, out, 0.0)
    return out if keepdim else out.view(tuple(d for i, d in enumerate(a.shape) if i not in dimensions))

def mean_dim(a, dim=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        if dtype not in _dtypes:
            raise NotImplementedError("RUDA mean supports float32, float16 and bfloat16")
        a = a.to(dtype)
    if a.dtype != torch.float32:
        return mean_dim(a.float(), dim, keepdim).to(a.dtype)
    result = sum_dim(a, dim, keepdim, dtype=dtype)
    dimensions = tuple(range(a.ndim)) if dim is None or len(dim) == 0 else tuple(_dimension(d, a.ndim) for d in dim)
    count = math.prod(a.shape[d] for d in dimensions) if a.ndim else 1
    return div(result, count)

def _norm_axes(a, normalized_shape, weight, bias):
    normalized_shape = tuple(normalized_shape)
    if not normalized_shape or len(normalized_shape) > a.ndim or tuple(a.shape[-len(normalized_shape):]) != normalized_shape:
        raise RuntimeError("normalized_shape must match the trailing input dimensions")
    for value in (weight, bias):
        if value is not None and (tuple(value.shape) != normalized_shape or value.device != a.device or value.dtype != a.dtype):
            raise RuntimeError("LayerNorm affine parameters must match normalized_shape, dtype and device")
    return tuple(range(a.ndim - len(normalized_shape), a.ndim))

def layer_norm(a, normalized_shape, weight=None, bias=None, eps=1e-5):
    axes = _norm_axes(a, normalized_shape, weight, bias)
    if a.dtype != torch.float32:
        result, mean, rstd = layer_norm(a.float(), normalized_shape,
            weight.float() if weight is not None else None, bias.float() if bias is not None else None, eps)
        return result.to(a.dtype), mean, rstd
    mean = mean_dim(a, axes, True)
    centered = a - mean
    variance = mean_dim(centered * centered, axes, True)
    rstd = torch.rsqrt(variance + eps)
    result = centered * rstd
    if weight is not None:
        result = result * weight
    if bias is not None:
        result = result + bias
    return result, mean, rstd

def layer_norm_backward(grad, a, normalized_shape, mean, rstd, weight, bias, output_mask):
    axes = _norm_axes(a, normalized_shape, weight, bias)
    if a.dtype != torch.float32:
        results = layer_norm_backward(grad.float(), a.float(), normalized_shape, mean, rstd,
            weight.float() if weight is not None else None, bias.float() if bias is not None else None, output_mask)
        return tuple(value.to(a.dtype) if value is not None else None for value in results)
    leading = tuple(range(a.ndim - len(axes)))
    normalized = (a - mean) * rstd
    dx = dw = db = None
    if output_mask[0]:
        scaled_grad = grad if weight is None else grad * weight
        dx = (scaled_grad - mean_dim(scaled_grad, axes, True)
              - normalized * mean_dim(scaled_grad * normalized, axes, True)) * rstd
    if output_mask[1] and weight is not None:
        dw = sum_dim(grad * normalized, leading) if leading else clone(grad * normalized)
    if output_mask[2] and bias is not None:
        db = sum_dim(grad, leading) if leading else clone(grad)
    return dx, dw, db

def pow_scalar(a, exponent):
    if exponent == 3:
        if a.dtype != torch.float32:
            return pow_scalar(a.float(), exponent).to(a.dtype)
        return mul(mul(a, a), a)
    if exponent == 2:
        return mul(a, a)
    if exponent == 1:
        return clone(a)
    if exponent == 0:
        return fill_(_out(a.shape, a), 1)
    raise NotImplementedError("RUDA native power currently supports exponents 0, 1, 2, 3")

def activation_backward(grad, output, *, tanh=False):
    if output.dtype == torch.float32:
        return _apply(18 if tanh else 16, grad, output)
    one = fill_(_out((), output), 1)
    if tanh:
        return mul(grad, add(one, mul(output, output), alpha=-1))
    return mul(mul(grad, add(one, output, alpha=-1)), output)

def clone(a, *, memory_format=torch.preserve_format):
    out = torch.empty_like(a, memory_format=memory_format)
    return out.copy_(a)

for name, function in {
    "fill_.Scalar": fill_, "zero_": lambda a: fill_(a, 0),
    "add.Tensor": add, "add.Scalar": add, "add_.Tensor": add_, "add_.Scalar": add_,
    "mul.Tensor": mul, "mul.Scalar": mul, "mul_.Tensor": mul_, "mul_.Scalar": mul_,
    "div.Tensor": div, "div.Scalar": div, "div.Tensor_mode": div, "div.Scalar_mode": div,
    "div_.Tensor": div_, "div_.Scalar": div_, "div_.Tensor_mode": div_, "div_.Scalar_mode": div_,
    "sub.Tensor": lambda a, b, alpha=1: add(a, b, alpha=-alpha),
    "sub.Scalar": lambda a, b, alpha=1: add(a, b, alpha=-alpha),
    "neg": lambda a: mul(a, -1),
    "mm": mm, "addmm": addmm, "bmm": bmm,
    "relu": lambda a: _apply(3, a),
    "threshold_backward": lambda grad, a, threshold: _apply(4, grad, a, scalar=float(threshold)),
    "sum": lambda a, dtype=None: sum_dim(a, dtype=dtype), "sum.dim_IntList": sum_dim,
    "mean": lambda a, dtype=None: mean_dim(a, dtype=dtype), "mean.dim": mean_dim,
    "exp": lambda a: _apply(9, a), "log": lambda a: _apply(10, a),
    "sqrt": lambda a: _apply(11, a), "rsqrt": lambda a: _apply(12, a),
    "sigmoid": lambda a: _apply(13, a), "sigmoid_backward": activation_backward,
    "silu": lambda a: _apply(14, a), "silu_": lambda a: _apply(14, a, out=a),
    "silu_backward": lambda grad, a: _apply(15, grad, a),
    "tanh": lambda a: _apply(17, a), "tanh_backward": lambda grad, y: activation_backward(grad, y, tanh=True),
    "_softmax": softmax, "_softmax_backward_data": softmax_backward,
    "_log_softmax": lambda a, dim, half_to_float=False: softmax(a, dim, half_to_float, logarithmic=True),
    "_log_softmax_backward_data": lambda grad, y, dim, dtype: softmax_backward(grad, y, dim, dtype, logarithmic=True),
    "native_layer_norm": layer_norm, "native_layer_norm_backward": layer_norm_backward,
    "pow.Tensor_Scalar": pow_scalar, "clone": clone,
}.items():
    _registry.impl(name, function)
