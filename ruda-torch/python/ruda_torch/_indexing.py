import torch
from . import _C
from ._ops import _registry, _dimension, _out, _typed_value, clone, div, fill_, masked_fill, mul


def _indices(a, index, *, empty_dtype=False):
    if index.device != a.device:
        raise RuntimeError("RUDA indices must be on the same device as the tensor")
    if not (empty_dtype and index.numel() == 0) and index.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("RUDA indices must have dtype int32 or int64")


def _nonempty_rank(a):
    return a.view(1) if a.ndim == 0 else a


def _prefix(a, shape, except_axis=None):
    for axis, size in enumerate(shape):
        if axis != except_axis and a.shape[axis] != size:
            a = a.narrow(axis, 0, size)
    return a


def gather(a, dim, index, *, sparse_grad=False):
    axis = _dimension(dim, a.ndim)
    _indices(a, index, empty_dtype=True)
    result = _out(index.shape, a)
    if index.numel() == 0:
        return result
    value, indices = _nonempty_rank(a), _nonempty_rank(index)
    if value.ndim != indices.ndim or any(indices.shape[d] > value.shape[d] for d in range(value.ndim) if d != axis):
        raise RuntimeError("gather index shape must fit the input outside the indexed dimension")
    value = _prefix(value, indices.shape, axis)
    _C.execute(102, value, indices, _nonempty_rank(result), float(axis))
    return result


def index_select(a, dim, index):
    axis = _dimension(dim, a.ndim)
    _indices(a, index)
    if index.ndim > 1:
        raise RuntimeError("index_select index must be a scalar or vector")
    if a.ndim == 0 and index.numel() != 1:
        raise RuntimeError("index_select of a scalar requires exactly one index")
    shape = list(a.shape)
    if shape:
        shape[axis] = index.numel()
    result = _out(shape, a)
    _C.execute(103, _nonempty_rank(a), index.reshape(-1), _nonempty_rank(result), float(axis))
    return result


def _scatter(a, dim, index, source, *, inplace=False, assign=False):
    axis = _dimension(dim, a.ndim)
    _indices(a, index, empty_dtype=True)
    tensor_source = isinstance(source, torch.Tensor)
    if tensor_source and (source.device != a.device or source.dtype != a.dtype):
        raise RuntimeError("scatter source must match the destination device and dtype")
    if inplace:
        _C.check_index_output(a, source if tensor_source else index, index)
    result = a if inplace else clone(a)
    if index.numel() == 0:
        return result
    indices, target = _nonempty_rank(index), _nonempty_rank(result)
    value = (_nonempty_rank(source) if tensor_source else
             _typed_value(source, a.dtype, a.device).expand(indices.shape))
    if indices.ndim != target.ndim or indices.ndim != value.ndim:
        raise RuntimeError("scatter index, source and destination must have matching ranks")
    if any(indices.shape[d] > value.shape[d] or d != axis and indices.shape[d] > target.shape[d]
           for d in range(indices.ndim)):
        raise RuntimeError("scatter index shape exceeds the source or destination")
    value = _prefix(value, indices.shape)
    target = _prefix(target, indices.shape, axis)
    _C.execute(106 if assign else 104, value, indices, target, float(axis))
    return result


def scatter_add(a, dim, index, source, *, inplace=False):
    return _scatter(a, dim, index, source, inplace=inplace)


def scatter(a, dim, index, source, *, inplace=False):
    return _scatter(a, dim, index, source, inplace=inplace, assign=True)


def index_add(a, dim, index, source, *, alpha=1, inplace=False):
    axis = _dimension(dim, a.ndim)
    _indices(a, index)
    if index.ndim > 1:
        raise RuntimeError("index_add index must be a scalar or vector")
    if source.device != a.device or source.dtype != a.dtype:
        raise RuntimeError("index_add source must match the destination device and dtype")
    value, target = _nonempty_rank(source), _nonempty_rank(a)
    if value.ndim != target.ndim or value.shape[axis] != index.numel():
        raise RuntimeError("index_add source dimension must match the number of indices")
    if any(value.shape[d] != target.shape[d] for d in range(target.ndim) if d != axis):
        raise RuntimeError("index_add source shape must match outside the indexed dimension")
    if inplace:
        _C.check_index_output(a, source, index)
    result = a if inplace else clone(a)
    if alpha != 1:
        value = mul(value, _typed_value(alpha, a.dtype, a.device))
    _C.execute(105, value, index.reshape(-1), _nonempty_rank(result), float(axis))
    return result


def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    if weight.ndim != 2:
        raise RuntimeError("embedding weight must be two-dimensional")
    return index_select(weight, 0, indices.reshape(-1)).view((*indices.shape, weight.shape[1]))


def embedding_dense_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq):
    _indices(grad, indices)
    if num_weights < 0 or grad.ndim != indices.ndim + 1 or grad.shape[:-1] != indices.shape:
        raise RuntimeError("invalid embedding gradient shape")
    if grad.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise RuntimeError("embedding gradients require a supported floating dtype")
    flat = indices.reshape(-1)
    width = grad.shape[-1]
    value = grad.float().reshape(flat.numel(), width)
    if scale_grad_by_freq:
        counts = fill_(torch.empty((num_weights,), device=grad.device, dtype=torch.int64), 0)
        ones = fill_(torch.empty(flat.shape, device=grad.device, dtype=torch.int64), 1)
        counts = index_add(counts, 0, flat, ones, inplace=True)
        frequencies = index_select(counts, 0, flat).float().view(-1, 1)
        value = div(value, frequencies)
    if padding_idx >= 0:
        value = masked_fill(value, (flat == padding_idx).view(-1, 1), 0)
    result = fill_(torch.empty((num_weights, width), device=grad.device, dtype=torch.float32), 0)
    return index_add(result, 0, flat, value, inplace=True).to(grad.dtype)


for name, function in {
    "gather": gather,
    "index_select": index_select,
    "scatter_add": scatter_add,
    "scatter_add_": lambda a, dim, index, source: scatter_add(a, dim, index, source, inplace=True),
    "scatter.src": scatter,
    "scatter.value": scatter,
    "scatter_.src": lambda a, dim, index, source: scatter(a, dim, index, source, inplace=True),
    "scatter_.value": lambda a, dim, index, value: scatter(a, dim, index, value, inplace=True),
    "index_add": index_add,
    "index_add_": lambda a, dim, index, source, alpha=1: index_add(a, dim, index, source, alpha=alpha, inplace=True),
    "embedding": embedding,
    "embedding_dense_backward": embedding_dense_backward,
}.items():
    _registry.impl(name, function)
