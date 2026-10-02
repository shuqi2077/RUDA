"""Native GPU output/gradient checks; missing hardware/API is a failure."""
import os

import pytest
import torch


@pytest.fixture(scope='module')
def r():
    assert os.environ.get('RUDA_CUDA_COMPILER') == 'ptx'
    import ruda_torch
    assert ruda_torch._sequence_available and ruda_torch._nf4_matmul_available
    return ruda_torch


@pytest.mark.parametrize('upper', [False, True])
@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('unit', [False, True])
def test_native_triangular_forward_and_gradients(r, upper, left, unit):
    torch.manual_seed(125)
    a = (torch.randn(1, 7, 7) * .1 + torch.eye(7)).requires_grad_()
    b = torch.randn((2, 7, 5) if left else (2, 5, 7), requires_grad=True)
    reference = torch.linalg.solve_triangular(a, b, upper=upper, left=left, unitriangular=unit)
    ad, bd = [x.detach().to('ruda').requires_grad_() for x in (a, b)]
    g = torch.randn_like(reference); gd = g.to('ruda')
    before = r.execution_stats()
    actual = r.solve_triangular(ad, bd, upper=upper, left=left, unitriangular=unit)
    actual.backward(gd); r.synchronize()
    after = r.execution_stats()
    for name in ('host_to_device_bytes', 'device_to_host_bytes'):
        assert after[name] == before[name]
    reference.backward(g)
    for got, want in ((actual.cpu(), reference), (ad.grad.cpu(), a.grad), (bd.grad.cpu(), b.grad)):
        torch.testing.assert_close(got, want, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize('native_forward', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('trainable', ['all', 'query'])
def test_native_delta_outputs_final_state_and_gradients(r, native_forward, dtype, trainable):
    torch.manual_seed(126)
    q = torch.randn(1, 2, 9, 5, dtype=dtype)
    k = torch.randn_like(q) * .15
    v = torch.randn(1, 2, 9, 7, dtype=dtype)
    beta = torch.rand(1, 2, 9); decay = -torch.rand_like(beta)
    state = torch.randn(1, 2, 5, 7)
    cpu = [x.requires_grad_(trainable == 'all' or i == 0) for i, x in enumerate((q, k, v, beta, decay, state))]
    device = [x.detach().to('ruda').requires_grad_(x.requires_grad) for x in cpu]
    # Independent per-token recurrence, not the production chunk algorithm.
    qs, ks, vs, bs, gs, ss = [x.float() for x in cpu]
    ys = []
    for t in range(9):
        ss = ss * gs[..., t, None, None].exp()
        residual = bs[..., t, None] * (vs[..., t, :] - (ks[..., t, :, None] * ss).sum(-2))
        ss = ss + ks[..., t, :, None] * residual.unsqueeze(-2)
        ys.append((qs[..., t, :, None] * ss).sum(-2) * .6)
    expected = torch.stack(ys, -2).to(dtype), ss
    gradients = [torch.randn_like(x) for x in expected]
    gd = [x.to('ruda') for x in gradients]
    before = r.execution_stats()
    actual = r.gated_delta_rule(*device[:5], initial_state=device[5], query_scale=.6,
                                chunk_size=4, native_forward=native_forward)
    ((actual[0] * gd[0]).sum() + (actual[1] * gd[1]).sum()).backward(); r.synchronize()
    after = r.execution_stats()
    for name in ('host_to_device_bytes', 'device_to_host_bytes'):
        assert after[name] == before[name]
    ((expected[0] * gradients[0]).sum() + (expected[1] * gradients[1]).sum()).backward()
    tol = {torch.float32: 4e-4, torch.float16: .007, torch.bfloat16: .045}[dtype]
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want, rtol=tol, atol=tol)
    for got, want in zip(device, cpu):
        if want.grad is None:
            assert got.grad is None
        else:
            torch.testing.assert_close(got.grad.cpu(), want.grad, rtol=tol, atol=tol)
