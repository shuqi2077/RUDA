import importlib
from types import SimpleNamespace
import pytest
import torch
from architecture_test_utils import NAME

seq = importlib.import_module(NAME + '.sequence_training')


@pytest.mark.parametrize('upper', [False, True])
@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('unit', [False, True])
def test_triangular_solve_broadcast_forward_and_gradients(upper, left, unit):
    torch.manual_seed(18)
    a = (torch.randn(1, 4, 4, dtype=torch.float64) * .1 + torch.eye(4, dtype=torch.float64)).requires_grad_()
    b = torch.randn((2, 4, 3) if left else (2, 3, 4), dtype=torch.float64, requires_grad=True)
    reference = torch.linalg.solve_triangular(a, b, upper=upper, left=left, unitriangular=unit)
    actual = seq.solve_triangular(a, b, upper=upper, left=left, unitriangular=unit)
    torch.testing.assert_close(actual, reference)
    grad = torch.randn_like(actual)
    got = torch.autograd.grad(actual, (a, b), grad)
    want = torch.autograd.grad(reference, (a, b), grad)
    for x, y in zip(got, want):
        torch.testing.assert_close(x, y)


def recurrent(q, k, v, beta, decay, state, scale):
    values = []
    for t in range(q.shape[-2]):
        state = state * decay[..., t, None, None].exp()
        residual = (v[..., t, :] - (state * k[..., t, :, None]).sum(-2)) * beta[..., t, None]
        state = state + k[..., t, :, None] * residual.unsqueeze(-2)
        values.append((state * q[..., t, :, None]).sum(-2) * scale)
    return torch.stack(values, -2), state


@pytest.mark.parametrize('chunk', [1, 3, 4, 16])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_gated_delta_full_recurrence_and_all_gradients(chunk, checkpoint):
    torch.manual_seed(19)
    q = torch.randn(2, 2, 7, 3, dtype=torch.float64)
    k = torch.randn_like(q) * .2
    v = torch.randn(2, 2, 7, 5, dtype=torch.float64)
    beta = torch.rand(2, 2, 7, dtype=torch.float64)
    decay = -torch.rand_like(beta)
    initial = torch.randn(2, 2, 3, 5, dtype=torch.float64)
    args = [t.requires_grad_() for t in (q, k, v, beta, decay, initial)]
    expected = recurrent(*args, .7)
    actual = seq.gated_delta_rule(*args[:5], initial_state=initial, query_scale=.7,
                                 chunk_size=chunk, checkpoint_chunks=checkpoint)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-11)
    gradient = [torch.randn_like(t) for t in actual]
    got = torch.autograd.grad(actual, args, gradient)
    want = torch.autograd.grad(expected, args, gradient)
    for x, y in zip(got, want):
        torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-10)


def test_delta_separate_chunks_preserve_initial_state_gradient():
    torch.manual_seed(20)
    q = torch.randn(1, 1, 9, 3) * .2
    v = torch.randn(1, 1, 9, 4)
    beta = torch.rand(1, 1, 9)
    decay = -torch.rand_like(beta) * .2
    initial = torch.randn(1, 1, 3, 4, requires_grad=True)
    y, state = seq.gated_delta_rule(q, q, v, beta, decay, initial_state=initial)
    first, middle = seq.gated_delta_rule(q[..., :4, :], q[..., :4, :], v[..., :4, :],
                                        beta[..., :4], decay[..., :4], initial_state=initial)
    last, final = seq.gated_delta_rule(q[..., 4:, :], q[..., 4:, :], v[..., 4:, :],
                                      beta[..., 4:], decay[..., 4:], initial_state=middle)
    torch.testing.assert_close(torch.cat((first, last), -2), y)
    torch.testing.assert_close(final, state)
    got = torch.autograd.grad(final.sum(), initial)[0]
    want = torch.autograd.grad(state.sum(), initial)[0]
    torch.testing.assert_close(got, want)


def test_strong_decay_does_not_overflow_unused_upper_triangle():
    q = torch.randn(1, 1, 5, 3, requires_grad=True)
    v = torch.randn(1, 1, 5, 4, requires_grad=True)
    beta = torch.ones(1, 1, 5)
    decay = torch.full_like(beta, -200.)
    y, final = seq.gated_delta_rule(q, q * .01, v, beta, decay)
    (y.sum() + final.sum()).backward()
    assert torch.isfinite(y).all() and torch.isfinite(q.grad).all() and torch.isfinite(v.grad).all()


@pytest.mark.parametrize('trainable', [(0,), (1,), (2,), (3,), (4,), (5,), (0, 2)])
@pytest.mark.parametrize('final_only', [False, True])
def test_native_backward_recompute_supports_partial_trainability(trainable, final_only):
    # Exercise the production native-autograd backward on explicit CPU inputs.
    # This does not replace, call or claim execution of the native forward.
    torch.manual_seed(137)
    q = torch.randn(1, 2, 5, 3)
    tensors = (q, q * .1, torch.randn(1, 2, 5, 4), torch.rand(1, 2, 5),
               -torch.rand(1, 2, 5), torch.randn(1, 2, 3, 4))
    inputs = [x.detach().requires_grad_(i in trainable) for i, x in enumerate(tensors)]
    output, final = recurrent(*inputs, .7)
    gy, gs = torch.randn_like(output), torch.randn_like(final)
    if final_only:
        gy.zero_()
    expected = torch.autograd.grad((output * gy).sum() + (final * gs).sum(),
                                   [inputs[i] for i in trainable], allow_unused=True)
    ctx = SimpleNamespace(saved_tensors=inputs, needs_input_grad=tuple(i in trainable for i in range(6)),
                          options=(.7, 3, True))
    gradients = seq._NativeDelta.backward(ctx, gy, gs)
    for i, value in enumerate(gradients[:6]):
        if i not in trainable:
            assert value is None
        else:
            wanted = expected[trainable.index(i)]
            if wanted is None:
                assert value is None
            else:
                torch.testing.assert_close(value, wanted, rtol=4e-5, atol=4e-6)
