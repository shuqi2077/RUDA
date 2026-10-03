"""Native factory/storage regressions. Host tensors are independent references only."""
import copy
import os

import pytest
import torch


@pytest.fixture(scope='module')
def r():
    assert os.environ.get('RUDA_CUDA_COMPILER') == 'ptx'
    import ruda_torch
    assert ruda_torch._factory_available
    return ruda_torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16,
                                   torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8])
@pytest.mark.parametrize('start,end,step', [(0, 17, 1), (13, 0, -3), (5, 5, 2)])
def test_arange_native_strides_and_no_transfers(r, dtype, start, end, step):
    expected = torch.arange(start, end, step, dtype=dtype)
    storage = torch.empty(expected.numel()*2+3, device='ruda', dtype=dtype).fill_(7)
    view = storage[1:1+2*expected.numel():2]
    before = r.execution_stats()
    actual = torch.arange(start, end, step, device='ruda', dtype=dtype)
    result = torch.arange(start, end, step, out=view)
    r.synchronize()
    after = r.execution_stats()
    assert result is view and view.storage_offset() == 1 and view.stride() == (2,)
    for key in ('host_to_device_bytes', 'device_to_host_bytes'):
        assert before[key] == after[key]
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(view.cpu(), expected, rtol=0, atol=0)
    assert torch.all(storage[::2].cpu() == 7)


def test_arange_large_integer_resize_and_errors(r):
    out = torch.empty(0, device='ruda', dtype=torch.int64)
    start = 2**60+3
    torch.arange(start, start+35, 3, out=out)
    torch.testing.assert_close(out.cpu(), torch.arange(start, start+35, 3), rtol=0, atol=0)
    for args in ((0, 5, 0), (5, 0, 1), (0, 5, -1), (float('nan'), 5, 1)):
        with pytest.raises(RuntimeError):
            torch.arange(*args, out=out)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_normal_generator_replay_strides_and_no_transfers(r, dtype):
    generator = torch.Generator(device='ruda').manual_seed(129)
    state = generator.get_state()
    out = torch.empty((17, 33), device='ruda', dtype=dtype).t()
    before = r.execution_stats()
    out.normal_(mean=.25, std=.5, generator=generator)
    generator.set_state(state)
    replay = torch.randn(out.shape, device='ruda', dtype=dtype, generator=generator)*.5+.25
    r.synchronize()
    after = r.execution_stats()
    for key in ('host_to_device_bytes', 'device_to_host_bytes'):
        assert before[key] == after[key]
    # The kernel applies the affine transform before storage narrowing.
    generator.set_state(state)
    exact = torch.empty(out.shape, device='ruda', dtype=dtype).normal_(.25, .5, generator=generator)
    torch.testing.assert_close(out.cpu(), exact.cpu(), rtol=0, atol=0)
    assert torch.isfinite(replay.cpu()).all()
    values = out.cpu().float()
    assert abs(values.mean().item()-.25) < .12 and abs(values.std().item()-.5) < .12
    r.manual_seed_all(731)
    saved = r.get_rng_state()
    a = torch.randn(29, device='ruda', dtype=dtype)
    r.set_rng_state(saved)
    b = torch.randn(29, device='ruda', dtype=dtype)
    torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)


def test_deepcopy_preserves_aliases_offsets_and_independent_storage(r):
    base = torch.arange(35, device='ruda', dtype=torch.float32).reshape(5, 7)
    before = r.execution_stats()
    cloned = copy.deepcopy({'a': base[:, 1::2], 'b': base.t()})
    r.synchronize()
    after = r.execution_stats()
    for key in ('host_to_device_bytes', 'device_to_host_bytes'):
        assert before[key] == after[key]
    assert cloned['a'].untyped_storage().data_ptr() == cloned['b'].untyped_storage().data_ptr()
    assert cloned['b'].untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
    assert cloned['a'].stride() == base[:, 1::2].stride() and cloned['a'].storage_offset() == 1
    torch.testing.assert_close(cloned['a'].cpu(), base[:, 1::2].cpu(), rtol=0, atol=0)
    cloned['b'].zero_()
    assert torch.count_nonzero(cloned['a'].cpu()) == 0
    assert torch.count_nonzero(base.cpu()) == 34
