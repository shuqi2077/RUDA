import torch
from ._ops import _registry
from .sequence_training import solve_triangular, triangular_mask


def _solve(a, b, *, upper, left=True, unitriangular=False):
    return solve_triangular(a, b, upper=upper, left=left, unitriangular=unitriangular)


def _solve_out(a, b, *, upper, left=True, unitriangular=False, out):
    result = _solve(a, b, upper=upper, left=left, unitriangular=unitriangular)
    if out.shape != result.shape or out.dtype != result.dtype or out.device != result.device:
        raise ValueError('triangular solve out must have the result shape/dtype/device')
    out.copy_(result)
    return out


_registry.impl('linalg_solve_triangular', _solve)
_registry.impl('linalg_solve_triangular.out', _solve_out)
_registry.impl('tril', lambda x, diagonal=0: triangular_mask(x, diagonal))
_registry.impl('triu', lambda x, diagonal=0: triangular_mask(x, diagonal, upper=True))
