import math
import torch
from torch._prims_common import suggest_memory_format
from . import _C

_registry = torch.library.Library("aten", "IMPL", "PrivateUse1")
_dtypes = (torch.float32, torch.float16, torch.bfloat16)
_storage_dtypes = _dtypes + (torch.bool, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8)

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
    _C.fill(tensor, value)
    return tensor

def _binary(op, a, b, *, scalar=0.0, inplace=False):
    if op in (1, 2) and (a.dtype not in _dtypes or isinstance(b, torch.Tensor) and b.dtype not in _dtypes):
        return _integral_arithmetic(op, a, b, alpha=scalar if op == 1 else 1, inplace=inplace)
    if a.dtype not in _dtypes:
        raise RuntimeError("RUDA arithmetic requires a supported floating dtype")
    dtype = a.dtype
    if isinstance(b, torch.Tensor):
        if b.device != a.device or b.dtype not in _dtypes:
            raise RuntimeError("RUDA arithmetic inputs must use supported floating dtypes on the same device")
        dtype = torch.promote_types(a.dtype, b.dtype)
    else:
        b = _scalar(b, a)
    left, right = torch.broadcast_tensors(a, b)
    if inplace:
        if a.shape != left.shape:
            raise RuntimeError("RUDA in-place output cannot change shape through broadcasting")
        result = a
    else:
        result = torch.empty(left.shape, device=a.device, dtype=dtype)
    _C.execute(op, left, right, result, float(scalar))
    return result


def add(a, b, *, alpha=1):
    return _binary(1, a, b, scalar=alpha)

def add_(a, b, *, alpha=1):
    return _binary(1, a, b, inplace=True, scalar=alpha)

def mul(a, b):
    return _binary(2, a, b)

def mul_(a, b):
    return _binary(2, a, b, inplace=True)

def _division_op(rounding_mode):
    if rounding_mode is None:
        return 8
    if rounding_mode == "trunc":
        return 57
    if rounding_mode == "floor":
        raise NotImplementedError("RUDA floor division is not implemented")
    raise RuntimeError("rounding_mode must be None, 'trunc' or 'floor'")


def div(a, b, *, rounding_mode=None):
    return _binary(_division_op(rounding_mode), a, b)

def div_(a, b, *, rounding_mode=None):
    return _binary(_division_op(rounding_mode), a, b, inplace=True)

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
    if a.dtype not in _dtypes or dtype is not None and dtype not in _dtypes:
        return _primitive_reduce(99, a, dim, keepdim, dtype=dtype)
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
    if isinstance(exponent, int) and not isinstance(exponent, bool) and a.dtype in _dtypes:
        indices = _typed_value(exponent, torch.int64, a.device).expand(a.shape)
        return _apply(96, a, indices)
    raise NotImplementedError("RUDA power supports integer scalar exponents for floating tensors; non-integral exponents are not implemented")

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


def leaky_relu(a, negative_slope=0.01):
    return _apply(41, a, scalar=float(negative_slope))


def leaky_relu_(a, negative_slope=0.01):
    return _apply(41, a, out=a, scalar=float(negative_slope))


def leaky_relu_backward(grad, a, negative_slope, self_is_result):
    if self_is_result and negative_slope < 0:
        raise RuntimeError("in-place leaky_relu backward does not support a negative slope")
    return _apply(42, grad, a, scalar=float(negative_slope))


def log_sigmoid_forward(a):
    return _apply(47, a), _out((0,), a)


def log_sigmoid_backward(grad, a, buffer):
    return _apply(48, grad, a)


def _ternary(op, a, b, c, *, value=1, inplace=False, dtype=None):
    for tensor in (a, b, c):
        if tensor.device != a.device or tensor.dtype not in _dtypes:
            raise RuntimeError("RUDA ternary inputs must use supported floating dtypes on the same device")
    if dtype is None:
        dtype = torch.promote_types(torch.promote_types(a.dtype, b.dtype), c.dtype)
    shape = torch.broadcast_shapes(a.shape, b.shape, c.shape)
    if inplace and a.shape != shape:
        raise RuntimeError("RUDA in-place output cannot change shape through broadcasting")
    if inplace:
        result = a
    else:
        result = torch.empty(shape, device=a.device, dtype=dtype)
        result.copy_(a)
    _C.execute(op, b.expand(shape), c.expand(shape), result, float(value))
    return result


def addcmul(a, tensor1, tensor2, *, value=1):
    return _ternary(28, a, tensor1, tensor2, value=value)


def addcmul_(a, tensor1, tensor2, *, value=1):
    return _ternary(28, a, tensor1, tensor2, value=value, inplace=True)


def addcdiv(a, tensor1, tensor2, *, value=1):
    return _ternary(29, a, tensor1, tensor2, value=value)


def addcdiv_(a, tensor1, tensor2, *, value=1):
    return _ternary(29, a, tensor1, tensor2, value=value, inplace=True)


def lerp(a, end, weight, *, inplace=False):
    _scalar(end, a)
    weight = _scalar(weight, a)
    return _ternary(49, a, end, weight, inplace=inplace, dtype=a.dtype)


def cat(tensors, dim=0):
    tensors = tuple(tensors)
    if not tensors:
        raise RuntimeError("cat expects a non-empty tensor list")
    first = tensors[0]
    dtype = first.dtype
    memory_format = None
    for tensor in tensors:
        if tensor.device != first.device or tensor.dtype not in _storage_dtypes or tensor.layout != torch.strided:
            raise RuntimeError("RUDA cat requires supported strided tensors on the same device")
        if tensor.ndim == 0:
            raise RuntimeError("zero-dimensional tensors cannot be concatenated")
        dtype = torch.promote_types(dtype, tensor.dtype)
        current_format = suggest_memory_format(tensor)
        if memory_format is None:
            memory_format = current_format
        elif memory_format != current_format:
            memory_format = torch.contiguous_format
    inputs = tuple(tensor for tensor in tensors if tensor.ndim != 1 or tensor.shape[0] != 0)
    if not inputs:
        return torch.empty((0,), device=first.device, dtype=dtype)
    dim = _dimension(dim, inputs[0].ndim)
    shape = list(inputs[0].shape)
    shape[dim] = 0
    for tensor in inputs:
        if tensor.ndim != len(shape) or any(
            tensor.shape[axis] != shape[axis] for axis in range(len(shape)) if axis != dim
        ):
            raise RuntimeError("cat tensor sizes must match except in the concatenation dimension")
        shape[dim] += tensor.shape[dim]
    result = torch.empty(shape, device=first.device, dtype=dtype, memory_format=memory_format)
    cursor = 0
    for tensor in inputs:
        if tensor.numel() != 0:
            destination = torch.as_strided(result, tensor.shape, result.stride(), cursor * result.stride(dim))
            destination.copy_(tensor)
        cursor += tensor.shape[dim]
    return result


def stack(tensors, dim=0):
    tensors = tuple(tensors)
    if not tensors:
        raise RuntimeError("stack expects a non-empty tensor list")
    shape = tensors[0].shape
    if any(tensor.shape != shape for tensor in tensors):
        raise RuntimeError("stack expects every tensor to have the same shape")
    dim = _dimension(dim, len(shape) + 1)
    if dim < len(shape):
        output_shape = list(shape)
        output_shape.insert(dim, len(tensors))
        return cat(tensors, dim).view(output_shape)
    return cat(tuple(tensor.unsqueeze(dim) for tensor in tensors), dim)


def mv(matrix, vector):
    if matrix.ndim != 2 or vector.ndim != 1 or matrix.shape[1] != vector.shape[0]:
        raise RuntimeError("mv requires a matrix and a compatible vector")
    return mm(matrix, vector.unsqueeze(1)).squeeze(1)


def dot(a, b):
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise RuntimeError("dot requires vectors with the same length")
    return mm(a.unsqueeze(0), b.unsqueeze(1)).view(())


def outer(a, b):
    if a.ndim != 1 or b.ndim != 1:
        raise RuntimeError("outer requires one-dimensional tensors")
    if a.device != b.device or a.dtype not in _dtypes or b.dtype not in _dtypes:
        raise RuntimeError("RUDA outer requires supported floating tensors on the same device")
    dtype = torch.promote_types(a.dtype, b.dtype)
    return mul(a.to(dtype).unsqueeze(1), b.to(dtype).unsqueeze(0))


def rsub(a, b, alpha=1):
    return _binary(50, a, b, scalar=alpha)


def constant_pad_nd(a, pad, value=0):
    pad = tuple(pad)
    if len(pad) % 2 or len(pad) > 2 * a.ndim:
        raise RuntimeError("padding must contain pairs for no more than the input rank")
    cropped = a
    for axis in range(a.ndim - len(pad) // 2, a.ndim):
        index = 2 * (a.ndim - axis - 1)
        left, right = pad[index:index + 2]
        if left < 0:
            cropped = cropped.narrow(axis, -left, cropped.shape[axis] + left)
        if right < 0:
            cropped = cropped.narrow(axis, 0, cropped.shape[axis] + right)
    if all(amount <= 0 for amount in pad):
        return clone(cropped)
    shape = list(a.shape)
    for axis in range(a.ndim - len(pad) // 2, a.ndim):
        index = 2 * (a.ndim - axis - 1)
        shape[axis] += pad[index] + pad[index + 1]
        if shape[axis] < 0:
            raise RuntimeError("negative padding results in a negative output size")
    result = torch.empty(shape, dtype=a.dtype, device=a.device, memory_format=suggest_memory_format(a))
    fill_(result, value)
    if cropped.numel() != 0:
        destination = result
        for axis in range(a.ndim - len(pad) // 2, a.ndim):
            index = 2 * (a.ndim - axis - 1)
            left, right = pad[index:index + 2]
            if left > 0:
                destination = destination.narrow(axis, left, destination.shape[axis] - left)
            if right > 0:
                destination = destination.narrow(axis, 0, destination.shape[axis] - right)
        destination.copy_(cropped)
    return result


def _matrix_bias(result, bias, beta, alpha):
    if alpha != 1:
        mul_(result, alpha)
    if beta != 0:
        add_(result, bias, alpha=beta)
    return result


def addmv(bias, matrix, vector, *, beta=1, alpha=1):
    _scalar(vector, matrix)
    _scalar(bias, matrix)
    if matrix.ndim != 2 or vector.ndim != 1 or matrix.shape[1] != vector.shape[0]:
        raise RuntimeError("addmv requires a matrix and a compatible vector")
    shape = (matrix.shape[0],)
    if torch.broadcast_shapes(bias.shape, shape) != shape:
        raise RuntimeError("addmv bias cannot broadcast to the output shape")
    if matrix.dtype != torch.float32:
        return addmv(bias.float(), matrix.float(), vector.float(), beta=beta, alpha=alpha).to(matrix.dtype)
    return _matrix_bias(mv(matrix, vector), bias, beta, alpha)


def _batch_matrix_bias(bias, a, b, beta, alpha, reduce_batch):
    _scalar(b, a)
    _scalar(bias, a)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise RuntimeError("batched matrix inputs must have compatible batch and matrix dimensions")
    shape = (a.shape[1], b.shape[2]) if reduce_batch else (a.shape[0], a.shape[1], b.shape[2])
    if torch.broadcast_shapes(bias.shape, shape) != shape:
        raise RuntimeError("batched matrix bias cannot broadcast to the output shape")
    if a.dtype != torch.float32:
        return _batch_matrix_bias(bias.float(), a.float(), b.float(), beta, alpha, reduce_batch).to(a.dtype)
    result = bmm(a, b)
    if reduce_batch:
        result = sum_dim(result, (0,))
    return _matrix_bias(result, bias, beta, alpha)


def addbmm(bias, a, b, *, beta=1, alpha=1):
    return _batch_matrix_bias(bias, a, b, beta, alpha, True)


def baddbmm(bias, a, b, *, beta=1, alpha=1):
    return _batch_matrix_bias(bias, a, b, beta, alpha, False)


def hardtanh(a, min_val=-1, max_val=1, *, inplace=False):
    return _apply(51, a, _scalar(max_val, a), scalar=float(min_val), out=a if inplace else None)


def hardtanh_backward(grad, a, min_val, max_val):
    lower = _apply(4, grad, a, scalar=float(min_val))
    return _apply(52, lower, a, scalar=float(max_val))


def softplus(a, beta=1, threshold=20):
    return _apply(53, a, _scalar(threshold, a), scalar=float(beta))


def softplus_backward(grad, a, beta, threshold):
    _scalar(grad, a)
    return _ternary(54, grad, a, _scalar(threshold, a), value=beta, dtype=a.dtype)


def mse_loss(a, target, reduction=1):
    _scalar(target, a)
    if reduction not in (0, 1, 2):
        raise RuntimeError("MSE reduction must be none, mean or sum")
    if a.dtype != torch.float32:
        return mse_loss(a.float(), target.float(), reduction).to(a.dtype)
    loss = _apply(55, a, target)
    if reduction == 0:
        return loss
    return mean_dim(loss) if reduction == 1 else sum_dim(loss)


def mse_loss_backward(grad, a, target, reduction):
    _scalar(target, a)
    _scalar(grad, a)
    if reduction not in (0, 1, 2):
        raise RuntimeError("MSE reduction must be none, mean or sum")
    norm = 2.0
    if reduction == 1:
        norm = 2.0 / a.numel() if a.numel() else float("inf")
    return _ternary(56, grad, a, target, value=norm, dtype=a.dtype)


def fill_tensor_(a, value):
    if value.ndim != 0:
        raise RuntimeError("fill_ only supports a zero-dimensional value tensor")
    return a.copy_(value)


def fill(a, value):
    result = torch.empty_like(a, memory_format=torch.preserve_format)
    if isinstance(value, torch.Tensor):
        return fill_tensor_(result, value)
    return fill_(result, value)


def hardshrink(a, lambd=0.5):
    boundary = _scalar(lambd, a).to(a.dtype)
    return _apply(58, a, boundary)


def softshrink(a, lambd=0.5):
    if not 0 <= lambd <= torch.finfo(a.dtype).max:
        raise RuntimeError("softshrink lambda must be nonnegative and within the input dtype range")
    boundary = _scalar(lambd, a).to(a.dtype)
    return _apply(59, a, boundary)


def shrink_backward(grad, a, lambd):
    _scalar(grad, a)
    boundary = _scalar(lambd, a).to(a.dtype)
    return _ternary(60, grad, a, boundary, dtype=a.dtype)


def threshold(a, threshold, value, *, inplace=False):
    boundary = _scalar(threshold, a).to(a.dtype)
    replacement = _scalar(value, a).to(a.dtype)
    result = _ternary(61, replacement, a, boundary, dtype=a.dtype)
    if inplace:
        a.copy_(result)
        return a
    return result


def prelu_kernel(a, weight):
    _scalar(weight, a)
    return _apply(62, a, weight)


def threshold_backward(grad, a, threshold):
    _scalar(grad, a)
    boundary = _scalar(threshold, a).to(a.dtype)
    return _ternary(73, grad, a, boundary, dtype=a.dtype)


def prelu_kernel_backward(grad, a, weight):
    _scalar(weight, a)
    _scalar(grad, a)
    a, weight, grad = torch.broadcast_tensors(a, weight, grad)
    grad_input = _ternary(63, grad, a, weight, dtype=a.dtype)
    grad_weight = _apply(64, a, grad)
    return grad_input, grad_weight


def _piecewise_loss(a, target, reduction, boundary, *, huber):
    if reduction not in (0, 1, 2):
        raise RuntimeError("loss reduction must be none, mean or sum")
    if huber:
        if not boundary > 0:
            raise RuntimeError("huber_loss delta must be positive")
    elif not boundary >= 0:
        raise RuntimeError("smooth_l1_loss beta must be nonnegative")
    if a.device != target.device or a.dtype not in _dtypes or target.dtype not in _dtypes:
        raise RuntimeError("RUDA loss inputs must use supported floating dtypes on the same device")
    dtype = torch.promote_types(a.dtype, target.dtype)
    loss = _apply(66 if huber else 65, a.to(dtype), target.to(dtype), scalar=float(boundary))
    if reduction == 0:
        return loss
    return mean_dim(loss) if reduction == 1 else sum_dim(loss)


def smooth_l1_loss(a, target, reduction=1, beta=1.0):
    return _piecewise_loss(a, target, reduction, beta, huber=False)


def huber_loss(a, target, reduction=1, delta=1.0):
    return _piecewise_loss(a, target, reduction, delta, huber=True)


def _piecewise_loss_backward(grad, a, target, reduction, boundary, *, huber):
    _scalar(target, a)
    _scalar(grad, a)
    if reduction not in (0, 1, 2):
        raise RuntimeError("loss reduction must be none, mean or sum")
    norm = 1.0
    if reduction == 1:
        norm = 1.0 / a.numel() if a.numel() else float("inf")
    normalization = _scalar(norm, a).to(a.dtype)
    difference = add(a, target, alpha=-1)
    return _ternary(68 if huber else 67, normalization, difference, grad,
                    value=boundary, dtype=a.dtype)


def smooth_l1_loss_backward(grad, a, target, reduction, beta):
    return _piecewise_loss_backward(grad, a, target, reduction, beta, huber=False)


def huber_loss_backward(grad, a, target, reduction, delta):
    return _piecewise_loss_backward(grad, a, target, reduction, delta, huber=True)


def _glu_halves(a, dim):
    if a.ndim == 0:
        raise RuntimeError("glu does not support zero-dimensional tensors")
    dim = _dimension(dim, a.ndim)
    if a.shape[dim] % 2:
        raise RuntimeError("glu requires an even size along the split dimension")
    width = a.shape[dim] // 2
    return a.narrow(dim, 0, width), a.narrow(dim, width, width), dim


def glu(a, dim=-1):
    left, gate, _ = _glu_halves(a, dim)
    if left.numel() == 0:
        return _out(left.shape, a)
    return _apply(71, left, gate)


def glu_backward(grad, a, dim):
    _scalar(grad, a)
    left, gate, dim = _glu_halves(a, dim)
    if grad.shape != left.shape:
        raise RuntimeError("glu gradient shape must match the forward output")
    if a.numel() == 0:
        return _out(a.shape, a)
    grad_left = _apply(71, grad, gate)
    grad_gate = _ternary(72, grad, left, gate, dtype=a.dtype)
    return cat((grad_left, grad_gate), dim)


def _adaptive_avg_pool(a, output_size, spatial_dims):
    output_size = tuple(output_size)
    if a.ndim not in (spatial_dims + 1, spatial_dims + 2):
        raise RuntimeError("adaptive average pooling input rank does not match its spatial dimensions")
    if len(output_size) != spatial_dims or any(size < 0 for size in output_size):
        raise RuntimeError("adaptive average pooling requires nonnegative spatial output sizes")
    if any(size == 0 for size in a.shape[-spatial_dims:]):
        raise RuntimeError("adaptive average pooling requires nonempty input spatial dimensions")
    if spatial_dims == 3 and any(size == 0 for size in a.shape[1:]):
        raise RuntimeError("adaptive_avg_pool3d requires nonempty non-batch dimensions")
    shape = tuple(a.shape[:-spatial_dims]) + output_size
    result = torch.empty(shape, device=a.device, dtype=a.dtype, memory_format=suggest_memory_format(a))
    if result.numel() != 0:
        _C.execute(74 if spatial_dims == 2 else 76, a, a, result, 0.0)
    return result


def adaptive_avg_pool2d(a, output_size):
    return _adaptive_avg_pool(a, output_size, 2)


def adaptive_avg_pool3d(a, output_size):
    return _adaptive_avg_pool(a, output_size, 3)


def _adaptive_avg_pool_backward(grad, a, spatial_dims):
    _scalar(grad, a)
    if a.ndim not in (spatial_dims + 1, spatial_dims + 2) or grad.ndim != a.ndim:
        raise RuntimeError("adaptive average pooling gradient rank must match the input")
    if grad.shape[:-spatial_dims] != a.shape[:-spatial_dims]:
        raise RuntimeError("adaptive average pooling gradient batch and channel dimensions must match the input")
    if any(size == 0 for size in grad.shape[1:]):
        raise RuntimeError("adaptive average pooling requires nonempty gradient non-batch dimensions")
    result = torch.empty_like(a, memory_format=torch.contiguous_format)
    if result.numel() != 0:
        _C.execute(75 if spatial_dims == 2 else 77, grad, grad, result, 0.0)
    return result


def adaptive_avg_pool2d_backward(grad, a):
    return _adaptive_avg_pool_backward(grad, a, 2)


def adaptive_avg_pool3d_backward(grad, a):
    return _adaptive_avg_pool_backward(grad, a, 3)


def _typed_value(value, dtype, device):
    if dtype not in _storage_dtypes:
        raise RuntimeError("RUDA does not support the promoted dtype")
    if isinstance(value, torch.Tensor):
        if value.device != device:
            raise RuntimeError("RUDA operands must be on the same device")
        return value.to(dtype)
    result = torch.empty((), dtype=dtype, device=device)
    return fill_(result, value)


def _comparison(op, a, b, *, inplace=False):
    dtype = torch.result_type(a, b)
    left = _typed_value(a, dtype, a.device)
    if isinstance(b, int) and dtype not in _dtypes:
        right = _typed_value(b, torch.int64, a.device).to(dtype)
    else:
        right = _typed_value(b, dtype, a.device)
    left, right = torch.broadcast_tensors(left, right)
    if inplace and left.shape != a.shape:
        raise RuntimeError("in-place comparison cannot change the input shape")
    result = torch.empty(left.shape, dtype=torch.bool, device=a.device)
    _C.execute(op, left, right, result, 0.0)
    if inplace:
        a.copy_(result)
        return a
    return result


def _logical(op, a, b=None, *, inplace=False):
    if a.dtype not in _storage_dtypes:
        raise RuntimeError("RUDA logical operations require a supported dtype")
    if b is None:
        b = a
    if b.device != a.device or b.dtype not in _storage_dtypes:
        raise RuntimeError("RUDA logical operands must use supported dtypes on the same device")
    left, right = torch.broadcast_tensors(a.to(torch.bool), b.to(torch.bool))
    if inplace and left.shape != a.shape:
        raise RuntimeError("in-place logical operation cannot change the input shape")
    result = torch.empty(left.shape, dtype=torch.bool, device=a.device)
    _C.execute(op, left, right, result, 0.0)
    if inplace:
        a.copy_(result)
        return a
    return result


def where(condition, a, b):
    if condition.dtype != torch.bool:
        raise RuntimeError("where requires a boolean condition")
    dtype = torch.result_type(a, b)
    left = _typed_value(a, dtype, condition.device)
    right = _typed_value(b, dtype, condition.device)
    condition, left, right = torch.broadcast_tensors(condition, left, right)
    result = torch.empty(left.shape, dtype=dtype, device=condition.device)
    result.copy_(right)
    _C.execute(84, condition, left, result, 0.0)
    return result


def masked_fill(a, mask, value, *, inplace=False):
    if mask.dtype != torch.bool or mask.device != a.device:
        raise RuntimeError("masked_fill requires a boolean mask on the input device")
    if isinstance(value, torch.Tensor) and value.ndim != 0:
        raise RuntimeError("masked_fill requires a zero-dimensional value tensor")
    shape = torch.broadcast_shapes(a.shape, mask.shape)
    if inplace and shape != a.shape:
        raise RuntimeError("in-place masked_fill cannot change the input shape")
    replacement = _typed_value(value, a.dtype, a.device)
    result = where(mask, replacement, a)
    if inplace:
        a.copy_(result)
        return a
    return result


def _primitive_operands(a, b):
    dtype = torch.result_type(a, b)
    left = _typed_value(a, dtype, a.device)
    right = (_typed_value(b, torch.int64, a.device).to(dtype)
             if isinstance(b, int) and dtype not in _dtypes else _typed_value(b, dtype, a.device))
    return (*torch.broadcast_tensors(left, right), dtype)


def _primitive_result(result, a, inplace):
    if inplace:
        if result.shape != a.shape or not torch.can_cast(result.dtype, a.dtype):
            raise RuntimeError("RUDA in-place result cannot change shape or cast to the input dtype")
        a.copy_(result)
        return a
    return result


def _integral_arithmetic(op, a, b, *, alpha=1, inplace=False):
    if inplace:
        _C.check_inplace(a, b if isinstance(b, torch.Tensor) else a)
    left, right, dtype = _primitive_operands(a, b)
    if op != 2:
        if isinstance(alpha, bool) and dtype != torch.bool:
            raise RuntimeError("Boolean alpha only supported for Boolean results")
        if dtype not in _dtypes and not isinstance(alpha, int):
            raise RuntimeError("For integral input tensors alpha must be integral")
    if inplace and (left.shape != a.shape or not torch.can_cast(dtype, a.dtype)):
        raise RuntimeError("RUDA in-place result cannot change shape or cast to the input dtype")
    if dtype in _dtypes:
        result = _binary(1 if op == 94 else op, left, right,
                         scalar=-alpha if op == 94 else alpha if op == 1 else 0)
    elif dtype == torch.bool:
        if op == 1 and not alpha:
            result = clone(left)
        else:
            result = _logical(86 if op == 1 else 85, left, right)
    else:
        if op != 2 and alpha != 1:
            factor = _typed_value(alpha, torch.int64, a.device).to(dtype)
            right = _apply(95, right, factor)
        result = _apply(93 if op == 1 else 94 if op == 94 else 95, left, right)
    return _primitive_result(result, a, inplace)


def sub(a, b, *, alpha=1, inplace=False):
    if a.dtype == torch.bool or isinstance(b, bool) or isinstance(b, torch.Tensor) and b.dtype == torch.bool:
        raise RuntimeError("Subtraction with a bool operand is not supported")
    if isinstance(alpha, bool):
        raise RuntimeError("Boolean alpha only supported for Boolean results")
    if a.dtype not in _dtypes or isinstance(b, torch.Tensor) and b.dtype not in _dtypes:
        return _integral_arithmetic(94, a, b, alpha=alpha, inplace=inplace)
    return _binary(1, a, b, scalar=-alpha, inplace=inplace)


def neg(a, *, inplace=False):
    if a.dtype == torch.bool:
        raise RuntimeError("Negation of a bool tensor is not supported")
    return _binary(2, a, -1, inplace=inplace)


def bitwise(op, a, b=None, *, inplace=False):
    if inplace:
        _C.check_inplace(a, b if isinstance(b, torch.Tensor) else a)
    if b is None:
        left, right, dtype = a, a, a.dtype
    else:
        left, right, dtype = _primitive_operands(a, b)
    if dtype not in _storage_dtypes[3:]:
        raise RuntimeError("RUDA bitwise operations require bool or integral tensors")
    if inplace and (left.shape != a.shape or not torch.can_cast(dtype, a.dtype)):
        raise RuntimeError("RUDA in-place result cannot change shape or cast to the input dtype")
    result = _logical(op - 4, left, right) if dtype == torch.bool else _apply(op, left, right)
    return _primitive_result(result, a, inplace)


def flip(a, dims):
    dimensions = tuple(_dimension(dim, a.ndim) for dim in dims)
    if len(set(dimensions)) != len(dimensions):
        raise RuntimeError("duplicate flip dimension")
    if not dimensions or a.ndim == 0:
        return clone(a)
    result = a
    for dim in dimensions:
        output = _out(a.shape, a)
        _C.execute(97, result, result, output, float(dim))
        result = output
    return result if a.is_contiguous() else torch.empty_like(a, memory_format=torch.preserve_format).copy_(result)


def _primitive_reduce(op, a, dim=None, keepdim=False, *, dtype=None):
    dtype = dtype or (a.dtype if a.dtype in _dtypes else torch.int64)
    if dtype not in _storage_dtypes or dtype == torch.bool:
        raise NotImplementedError("RUDA primitive reduction requires a supported numeric output dtype")
    if dim is None or isinstance(dim, (tuple, list)) and not dim:
        dimensions = tuple(range(a.ndim))
    else:
        dimensions = tuple(_dimension(d, a.ndim) for d in (dim if isinstance(dim, (tuple, list)) else (dim,)))
        if len(set(dimensions)) != len(dimensions):
            raise RuntimeError("duplicate reduction dimension")
    result = a.to(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        result = result.float()
    if a.ndim == 0:
        return clone(result).to(dtype)
    for axis in dimensions:
        shape = list(result.shape)
        shape[axis] = 1
        output = _out(shape, result)
        if result.shape[axis] == 0:
            fill_(output, 1 if op == 98 else 0)
        else:
            _C.execute(op, result, result, output, float(axis))
        result = output
    if not keepdim:
        result = result.view(tuple(size for axis, size in enumerate(a.shape) if axis not in dimensions))
    return result.to(dtype)


def prod(a, dim=None, keepdim=False, *, dtype=None):
    return _primitive_reduce(98, a, dim, keepdim, dtype=dtype)


def _boolean_reduce(a, dim=None, keepdim=False, *, every):
    result = _primitive_reduce(98 if every else 99, a.to(torch.bool).to(torch.int64), dim, keepdim)
    return result.to(torch.bool).to(torch.uint8 if a.dtype == torch.uint8 else torch.bool)


def _cumulative(op, a, dim, *, dtype=None):
    axis = _dimension(dim, a.ndim)
    dtype = dtype or (a.dtype if a.dtype in _dtypes else torch.int64)
    if dtype not in _storage_dtypes or dtype == torch.bool:
        raise NotImplementedError("RUDA cumulative operations require a supported numeric output dtype")
    value = a.to(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        value = value.float()
    if a.ndim == 0:
        return clone(value).to(dtype)
    return _apply(op, value, scalar=float(axis)).to(dtype)


for name, function in {
    "fill_.Scalar": fill_, "zero_": lambda a: fill_(a, 0),
    "fill_.Tensor": fill_tensor_, "fill.Scalar": fill, "fill.Tensor": fill,
    "zero": lambda a: fill(a, 0),
    "where.self": where, "where.ScalarSelf": where, "where.ScalarOther": where, "where.Scalar": where,
    "masked_fill.Scalar": masked_fill, "masked_fill.Tensor": masked_fill,
    "masked_fill_.Scalar": lambda a, mask, value: masked_fill(a, mask, value, inplace=True),
    "masked_fill_.Tensor": lambda a, mask, value: masked_fill(a, mask, value, inplace=True),
    "logical_and": lambda a, b: _logical(85, a, b),
    "logical_or": lambda a, b: _logical(86, a, b),
    "logical_xor": lambda a, b: _logical(87, a, b),
    "logical_not": lambda a: _logical(88, a),
    "logical_and_": lambda a, b: _logical(85, a, b, inplace=True),
    "logical_or_": lambda a, b: _logical(86, a, b, inplace=True),
    "logical_xor_": lambda a, b: _logical(87, a, b, inplace=True),
    "logical_not_": lambda a: _logical(88, a, inplace=True),
    "add.Tensor": add, "add.Scalar": add, "add_.Tensor": add_, "add_.Scalar": add_,
    "mul.Tensor": mul, "mul.Scalar": mul, "mul_.Tensor": mul_, "mul_.Scalar": mul_,
    "addcmul": addcmul, "addcmul_": addcmul_, "addcdiv": addcdiv, "addcdiv_": addcdiv_,
    "lerp.Scalar": lerp, "lerp.Tensor": lerp,
    "lerp_.Scalar": lambda a, end, weight: lerp(a, end, weight, inplace=True),
    "lerp_.Tensor": lambda a, end, weight: lerp(a, end, weight, inplace=True),
    "div.Tensor": div, "div.Scalar": div, "div.Tensor_mode": div, "div.Scalar_mode": div,
    "div_.Tensor": div_, "div_.Scalar": div_, "div_.Tensor_mode": div_, "div_.Scalar_mode": div_,
    "sub.Tensor": sub, "sub.Scalar": sub,
    "sub_.Tensor": lambda a, b, alpha=1: sub(a, b, alpha=alpha, inplace=True),
    "sub_.Scalar": lambda a, b, alpha=1: sub(a, b, alpha=alpha, inplace=True),
    "rsub.Tensor": rsub, "rsub.Scalar": rsub,
    "neg": neg,
    "neg_": lambda a: neg(a, inplace=True),
    "mm": mm, "addmm": addmm, "bmm": bmm,
    "mv": mv, "dot": dot, "outer": outer, "cat": cat, "stack": stack,
    "ger": outer, "vdot": dot,
    "addmv": addmv, "addbmm": addbmm, "baddbmm": baddbmm,
    "constant_pad_nd": constant_pad_nd,
    "_adaptive_avg_pool2d": adaptive_avg_pool2d, "_adaptive_avg_pool2d_backward": adaptive_avg_pool2d_backward,
    "_adaptive_avg_pool3d": adaptive_avg_pool3d, "_adaptive_avg_pool3d_backward": adaptive_avg_pool3d_backward,
    "relu": lambda a: _apply(3, a),
    "relu_": lambda a: _apply(3, a, out=a),
    "threshold_backward": threshold_backward,
    "sum": lambda a, dtype=None: sum_dim(a, dtype=dtype), "sum.dim_IntList": sum_dim,
    "mean": lambda a, dtype=None: mean_dim(a, dtype=dtype), "mean.dim": mean_dim,
    "exp": lambda a: _apply(9, a), "log": lambda a: _apply(10, a),
    "exp_": lambda a: _apply(9, a, out=a), "log_": lambda a: _apply(10, a, out=a),
    "sqrt": lambda a: _apply(11, a), "rsqrt": lambda a: _apply(12, a),
    "sqrt_": lambda a: _apply(11, a, out=a), "rsqrt_": lambda a: _apply(12, a, out=a),
    "sigmoid": lambda a: _apply(13, a), "sigmoid_backward": activation_backward,
    "sigmoid_": lambda a: _apply(13, a, out=a),
    "silu": lambda a: _apply(14, a), "silu_": lambda a: _apply(14, a, out=a),
    "silu_backward": lambda grad, a: _apply(15, grad, a),
    "tanh": lambda a: _apply(17, a), "tanh_backward": lambda grad, y: activation_backward(grad, y, tanh=True),
    "tanh_": lambda a: _apply(17, a, out=a),
    "sin": lambda a: _apply(19, a), "sin_": lambda a: _apply(19, a, out=a),
    "cos": lambda a: _apply(20, a), "cos_": lambda a: _apply(20, a, out=a),
    "abs": lambda a: _apply(21, a), "abs_": lambda a: _apply(21, a, out=a),
    "sign": lambda a: _apply(22, a), "sign_": lambda a: _apply(22, a, out=a),
    "sgn": lambda a: _apply(22, a), "sgn_": lambda a: _apply(22, a, out=a),
    "floor": lambda a: _apply(23, a), "floor_": lambda a: _apply(23, a, out=a),
    "ceil": lambda a: _apply(24, a), "ceil_": lambda a: _apply(24, a, out=a),
    "trunc": lambda a: _apply(25, a), "trunc_": lambda a: _apply(25, a, out=a),
    "round": lambda a: _apply(26, a), "round_": lambda a: _apply(26, a, out=a),
    "reciprocal": lambda a: _apply(27, a), "reciprocal_": lambda a: _apply(27, a, out=a),
    "log1p": lambda a: _apply(35, a), "log1p_": lambda a: _apply(35, a, out=a),
    "sinh": lambda a: _apply(36, a), "sinh_": lambda a: _apply(36, a, out=a),
    "cosh": lambda a: _apply(37, a), "cosh_": lambda a: _apply(37, a, out=a),
    "asinh": lambda a: _apply(38, a), "acosh": lambda a: _apply(39, a), "atanh": lambda a: _apply(40, a),
    "leaky_relu": leaky_relu, "leaky_relu_": leaky_relu_, "leaky_relu_backward": leaky_relu_backward,
    "hardsigmoid": lambda a: _apply(43, a), "hardsigmoid_": lambda a: _apply(43, a, out=a),
    "hardsigmoid_backward": lambda grad, a: _apply(44, grad, a),
    "hardswish": lambda a: _apply(45, a), "hardswish_": lambda a: _apply(45, a, out=a),
    "hardswish_backward": lambda grad, a: _apply(46, grad, a),
    "hardtanh": hardtanh, "hardtanh_": lambda a, min_val=-1, max_val=1: hardtanh(a, min_val, max_val, inplace=True),
    "hardtanh_backward": hardtanh_backward,
    "softplus": softplus, "softplus_backward": softplus_backward,
    "hardshrink": hardshrink, "softshrink": softshrink,
    "hardshrink_backward": shrink_backward, "softshrink_backward": shrink_backward,
    "threshold": threshold,
    "threshold_": lambda a, threshold_value, value: threshold(a, threshold_value, value, inplace=True),
    "_prelu_kernel": prelu_kernel, "_prelu_kernel_backward": prelu_kernel_backward,
    "mish": lambda a: _apply(69, a), "mish_": lambda a: _apply(69, a, out=a),
    "mish_backward": lambda grad, a: _apply(70, grad, a),
    "glu": glu, "glu_backward": glu_backward,
    "mse_loss": mse_loss, "mse_loss_backward": mse_loss_backward,
    "smooth_l1_loss": smooth_l1_loss, "smooth_l1_loss_backward": smooth_l1_loss_backward,
    "huber_loss": huber_loss, "huber_loss_backward": huber_loss_backward,
    "log_sigmoid_forward": log_sigmoid_forward, "log_sigmoid_backward": log_sigmoid_backward,
    "_softmax": softmax, "_softmax_backward_data": softmax_backward,
    "_log_softmax": lambda a, dim, half_to_float=False: softmax(a, dim, half_to_float, logarithmic=True),
    "_log_softmax_backward_data": lambda grad, y, dim, dtype: softmax_backward(grad, y, dim, dtype, logarithmic=True),
    "native_layer_norm": layer_norm, "native_layer_norm_backward": layer_norm_backward,
    "pow.Tensor_Scalar": pow_scalar, "clone": clone,
    "flip": flip, "prod": prod, "prod.dim_int": prod,
    "all": lambda a: _boolean_reduce(a, every=True),
    "all.dim": lambda a, dim, keepdim=False: _boolean_reduce(a, dim, keepdim, every=True),
    "all.dims": lambda a, dim=None, keepdim=False: _boolean_reduce(a, dim, keepdim, every=True),
    "any": lambda a: _boolean_reduce(a, every=False),
    "any.dim": lambda a, dim, keepdim=False: _boolean_reduce(a, dim, keepdim, every=False),
    "any.dims": lambda a, dim=None, keepdim=False: _boolean_reduce(a, dim, keepdim, every=False),
    "cumsum": lambda a, dim, dtype=None: _cumulative(100, a, dim, dtype=dtype),
    "cumprod": lambda a, dim, dtype=None: _cumulative(101, a, dim, dtype=dtype),
    "bitwise_not": lambda a: bitwise(92, a),
    "bitwise_not_": lambda a: bitwise(92, a, inplace=True),
}.items():
    _registry.impl(name, function)


for name, op in (("eq", 78), ("ne", 79), ("lt", 80), ("le", 81), ("gt", 82), ("ge", 83)):
    for overload in ("Tensor", "Scalar"):
        _registry.impl(f"{name}.{overload}", lambda a, b, op=op: _comparison(op, a, b))
        _registry.impl(f"{name}_.{overload}", lambda a, b, op=op: _comparison(op, a, b, inplace=True))

for name, op in (("bitwise_and", 89), ("bitwise_or", 90), ("bitwise_xor", 91)):
    for overload in ("Tensor", "Scalar"):
        _registry.impl(f"{name}.{overload}", lambda a, b, op=op: bitwise(op, a, b))
        _registry.impl(f"{name}_.{overload}", lambda a, b, op=op: bitwise(op, a, b, inplace=True))
    _registry.impl(f"{name}.Scalar_Tensor", lambda a, b, op=op: bitwise(op, b, a))

from . import _indexing
