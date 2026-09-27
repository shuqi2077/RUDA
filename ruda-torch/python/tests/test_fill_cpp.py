"""Real C++ scalar binding and ABI payload tests; no GPU calculation is performed."""
import sys

import pytest
import torch

from test_v15_cpp import bridge


@pytest.mark.parametrize('dtype,value', [
    (torch.float32, 0), (torch.float32, 1.25), (torch.float32, -0.0),
    (torch.float32, float('nan')), (torch.float32, float('inf')),
    (torch.float32, float('-inf')), (torch.float16, 1.25),
    (torch.float16, -0.0), (torch.bfloat16, 1.25),
    (torch.bool, True), (torch.bool, False),
    (torch.int64, 2**60+3), (torch.int64, -(2**63)),
    (torch.int64, 2**63-1), (torch.int32, 2**31-1),
    (torch.int16, -32768), (torch.int8, -128), (torch.uint8, 255), (torch.uint8, -1),
    (torch.int64, 1.75), (torch.float32, complex(1.25, 0)),
])
def test_fill_scalar_preserves_native_payload(bridge, dtype, value):
    cpp, state, _ = bridge
    reference = torch.empty(1, dtype=dtype).fill_(value)
    expected = int.from_bytes(bytes(reference.view(torch.uint8).tolist()), sys.byteorder)
    out = torch.empty(1, dtype=dtype, device='ruda')
    before = len(state.fills)
    with pytest.raises(RuntimeError, match='explicit ABI test failure'):
        cpp.fill(out, value)
    assert state.fills[before:] == [expected]


@pytest.mark.parametrize('dtype,value', [
    (torch.int8, 128), (torch.uint8, 256), (torch.int64, 2**64),
    (torch.float32, complex(1, 2)), (torch.int64, float('inf')),
])
def test_fill_scalar_rejects_invalid_conversion_before_native(bridge, dtype, value):
    cpp, state, _ = bridge
    with pytest.raises((RuntimeError, OverflowError)):
        torch.empty(1, dtype=dtype).fill_(value)
    out = torch.empty(1, dtype=dtype, device='ruda')
    before = len(state.fills)
    with pytest.raises((RuntimeError, OverflowError)):
        cpp.fill(out, value)
    assert len(state.fills) == before


@pytest.mark.parametrize('value', [None, '1', [1]])
def test_fill_scalar_rejects_non_numbers(bridge, value):
    cpp, state, _ = bridge
    out = torch.empty(1, device='ruda')
    before = len(state.fills)
    with pytest.raises(TypeError):
        cpp.fill(out, value)
    assert len(state.fills) == before
