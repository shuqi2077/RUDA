"""Opaque native training operators with shape-only FakeTensor contracts."""
import torch


@torch.library.custom_op('ruda::nf4_decode', mutates_args=())
def nf4_decode(packed: torch.Tensor, scales: torch.Tensor, table: torch.Tensor,
               begin: int, rows: int, width: int, block: int, dtype: int) -> torch.Tensor:
    from . import _C
    return _C.nf4_decode(packed, scales, table, begin, rows, width, block, dtype)


@nf4_decode.register_fake
def _nf4_decode_fake(packed, scales, table, begin, rows, width, block, dtype):
    return packed.new_empty((rows, width), dtype=(torch.float32, torch.float16, torch.bfloat16)[dtype])


@torch.library.custom_op('ruda::mixed_mm', mutates_args=())
def mixed_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from . import _C
    result = a.new_empty((a.shape[0], b.shape[1]), dtype=torch.float32)
    _C.execute(7, a, b, result, 0.)
    return result


@mixed_mm.register_fake
def _mixed_mm_fake(a, b):
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.float32)


@torch.library.custom_op('ruda::nf4_matmul', mutates_args=())
def nf4_matmul(input: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
               table: torch.Tensor, columns: int, width: int, block: int,
               backward: bool) -> torch.Tensor:
    from . import _C
    return _C.nf4_matmul(input, packed, scales, table, columns, width, block, backward)


@nf4_matmul.register_fake
def _nf4_fake(input, packed, scales, table, columns, width, block, backward):
    return input.new_empty((input.shape[0], width if backward else columns),
                           dtype=torch.float32 if backward else input.dtype)


@torch.library.custom_op('ruda::triangular_solve', mutates_args=())
def triangular_solve(a: torch.Tensor, b: torch.Tensor, upper: bool, unit: bool) -> torch.Tensor:
    from . import _C
    return _C.triangular_solve(a, b, upper, unit)


@triangular_solve.register_fake
def _triangular_fake(a, b, upper, unit):
    return torch.empty_like(b)


@torch.library.custom_op('ruda::delta_forward', mutates_args=())
def delta_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  beta: torch.Tensor, decay: torch.Tensor, initial: torch.Tensor,
                  scale: float, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    from . import _C
    return _C.delta_forward(q, k, v, beta, decay, initial, scale, chunk)


@delta_forward.register_fake
def _delta_fake(q, k, v, beta, decay, initial, scale, chunk):
    return torch.empty_like(v), torch.empty_like(initial)
