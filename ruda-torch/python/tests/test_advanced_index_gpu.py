import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def backend():
    if os.environ.get("RUDA_REQUIRE_GPU") != "1":
        pytest.skip("requires the native RUDA GPU backend")
    import ruda_torch
    assert ruda_torch._C.abi_version == 10
    assert os.environ["RUDA_CUDA_COMPILER"] == "ptx"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bool, torch.int64])
@pytest.mark.parametrize("separated", [False, True])
def test_broadcast_advanced_index(dtype, separated):
    source = torch.arange(3 * 4 * 5).reshape(3, 4, 5).transpose(0, 2).to(dtype)
    first = torch.tensor([[0], [-1]], dtype=torch.int64)
    second = torch.tensor([[0, 1, -1]], dtype=torch.int32)
    indices = [first, None, second] if separated else [None, first, second]
    expected = torch.ops.aten.index.Tensor(source, indices)
    actual = torch.ops.aten.index.Tensor(source.to("ruda"), [x.to("ruda") if x is not None else None for x in indices])
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_index_out_resizes_and_reuses_output():
    source = torch.tensor([[True, False, True], [False, True, False]])
    a, b = torch.tensor([0, 1]), torch.tensor([2, 1])
    out = torch.empty(0, dtype=torch.bool, device="ruda")
    result = torch.ops.aten.index.Tensor_out(source.to("ruda"), [a.to("ruda"), b.to("ruda")], out=out)
    assert result is out
    torch.testing.assert_close(out.cpu(), source[a, b])


def test_advanced_index_bounds_are_checked_per_axis():
    source = torch.zeros((2, 3), device="ruda")
    # The flattened sum is in bounds, but the second axis index is not.
    with pytest.raises(RuntimeError):
        torch.ops.aten.index.Tensor(source, [torch.tensor([0]).to("ruda"), torch.tensor([3]).to("ruda")])


def test_advanced_index_vmap_mask():
    source = torch.tensor([[True, False, True], [False, True, False]])
    a, b = torch.tensor([0, 1]), torch.tensor([2, 1])
    expected = torch.vmap(lambda i, j: torch.ops.aten.index.Tensor(source, [i, j]))(a, b)
    device_source = source.to("ruda")
    actual = torch.vmap(lambda i, j: torch.ops.aten.index.Tensor(device_source, [i, j]))(a.to("ruda"), b.to("ruda"))
    torch.testing.assert_close(actual.cpu(), expected)
