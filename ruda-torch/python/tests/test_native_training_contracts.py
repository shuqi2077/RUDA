"""FakeTensor contracts only: never evidence of native device execution."""
import importlib
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from architecture_test_utils import NAME

native = importlib.import_module(NAME + '._native_training_ops')


@pytest.mark.parametrize('backward', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_nf4_opaque_fake_shape_and_accumulator_dtype(backward, dtype):
    with FakeTensorMode():
        x = torch.empty(7, 19 if backward else 33, dtype=dtype)
        result = native.nf4_matmul(x, torch.empty(314, dtype=torch.uint8), torch.empty(10),
                                   torch.empty(16), 19, 33, 64, backward)
        assert result.shape == (7, 33 if backward else 19)
        assert result.dtype == (torch.float32 if backward else dtype)
    assert NAME + '._C' not in sys.modules


def test_delta_and_triangular_opaque_fake_contracts():
    with FakeTensorMode():
        q = torch.empty(2, 3, 9, 5); v = torch.empty(2, 3, 9, 7)
        state = torch.empty(2, 3, 5, 7); decay = torch.empty(2, 3, 9)
        y, final = native.delta_forward(q, q, v, decay, decay, state, .5, 4)
        assert y.shape == v.shape and final.shape == state.shape
        a = torch.empty(2, 3, 8, 8); b = torch.empty(2, 3, 8, 12)
        assert native.triangular_solve(a, b, False, True).shape == b.shape
    assert NAME + '._C' not in sys.modules
