"""Real PyTorch AOT forward/backward tests on explicit CPU reference execution.

No RUDA extension is loaded and no native GPU execution is claimed by this file.
"""
import copy
import importlib
import json
from pathlib import Path
import sys
import types

import pytest
import torch
from torch.utils._pytree import tree_flatten

NAME = 'ruda_model_compile_test'
package = types.ModuleType(NAME)
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'ruda_torch')]
package._graph_available = False
sys.modules[NAME] = package
compiler = importlib.import_module(NAME + '.compiler')


@pytest.fixture(autouse=True)
def isolated_dynamo():
    torch._dynamo.reset()
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch._dynamo.reset()
    torch.set_num_threads(previous)


def wrap(model, **kwargs):
    kwargs.setdefault('native', 'off')
    return compiler.compile(model, device_type='cpu', **kwargs)


def clone_tree(values):
    return torch.utils._pytree.tree_map(
        lambda x: x.detach().clone().requires_grad_(x.requires_grad) if isinstance(x, torch.Tensor) else x,
        values,
    )


def assert_tree(actual, expected, **tolerances):
    a, a_spec = tree_flatten(actual)
    e, e_spec = tree_flatten(expected)
    assert a_spec == e_spec
    for x, y in zip(a, e):
        if isinstance(x, torch.Tensor):
            torch.testing.assert_close(x, y, **tolerances)
        else:
            assert x == y


def scalar_loss(value):
    values, _ = tree_flatten(value)
    return sum(x.square().mean() for x in values
               if isinstance(x, torch.Tensor) and x.requires_grad and x.numel())


class Residual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(8, 8)
        self.right = torch.nn.Linear(8, 8)
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, x):
        return self.norm(x + self.right(torch.nn.functional.gelu(self.left(x))))


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.TransformerEncoderLayer(8, 2, 16, dropout=.2, batch_first=True)

    def forward(self, x):
        return self.block(x)


class Recurrent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = torch.nn.GRUCell(8, 8)

    def forward(self, x):
        h = torch.zeros_like(x[:, 0])
        for i in range(x.shape[1]):
            h = self.cell(x[:, i], h)
        return h


class SharedEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 8)
        self.projection = torch.nn.Linear(8, 13, bias=False)
        self.projection.weight = self.embedding.weight

    def forward(self, tokens):
        return self.projection(self.embedding(tokens)).mean(1)


def model_case(name):
    if name == 'mlp':
        return torch.nn.Sequential(torch.nn.Linear(8, 12), torch.nn.SiLU(), torch.nn.Linear(12, 4)), torch.randn(3, 8, requires_grad=True)
    if name == 'residual':
        return Residual(), torch.randn(3, 8, requires_grad=True)
    if name == 'cnn':
        return CNN(), torch.randn(2, 3, 5, 5, requires_grad=True)
    if name == 'attention':
        return Attention(), torch.randn(2, 3, 8, requires_grad=True)
    if name == 'recurrent':
        return Recurrent(), torch.randn(2, 3, 8, requires_grad=True)
    if name == 'embedding':
        return SharedEmbedding(), torch.tensor([[1, 1, 2], [0, 3, 4]])
    raise AssertionError(name)


@pytest.mark.parametrize('name', ['mlp', 'residual', 'cnn', 'attention', 'recurrent', 'embedding'])
@pytest.mark.parametrize('native', ['off', 'auto'])
def test_models_forward_backward_and_optimizer_match(name, native):
    torch.manual_seed(103)
    model, x = model_case(name)
    reference = copy.deepcopy(model)
    compiled = wrap(model, native=native, min_native_ops=1)
    actual_optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.2)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01, momentum=.2)
    for step in range(3):
        actual_optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual_x, expected_x = clone_tree(x), clone_tree(x)
        torch.manual_seed(850 + step)
        expected = reference(expected_x)
        scalar_loss(expected).backward()
        torch.manual_seed(850 + step)
        actual = compiled(actual_x)
        scalar_loss(actual).backward()
        assert_tree(actual, expected, rtol=2e-4, atol=2e-5)
        if actual_x.requires_grad:
            torch.testing.assert_close(actual_x.grad, expected_x.grad, rtol=4e-4, atol=3e-5)
        for a, e in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, e.grad, rtol=4e-4, atol=3e-5)
        actual_optimizer.step()
        reference_optimizer.step()
        assert_tree(model.state_dict(), reference.state_dict(), rtol=2e-4, atol=2e-5)
    info = compiled.info
    assert info['forward_calls'] >= 3 and info['backward_calls'] >= 3
    assert any(g['phase'] == 'backward' for g in info['graphs'])
    assert info['native_replays'] == 0
    assert info['implicit_cpu_fallback'] is False
    json.dumps(info)
    compiled.close()


def test_nested_inputs_kwargs_outputs_and_non_tensor_values():
    def fn(batch, *, scale, extra=None):
        x, y = batch['tensors']
        z = (x + y) * scale
        return {'prediction': z.sin(), 'extras': [z, None, extra], 'shape': x.shape[0]}
    compiled = wrap(fn)
    args = {'tensors': [torch.randn(2, 3, requires_grad=True), torch.randn(1, 3, requires_grad=True)]}
    refs = clone_tree(args)
    got, ref = compiled(args, scale=2., extra='label'), fn(refs, scale=2., extra='label')
    assert_tree(got, ref)
    scalar_loss(got).backward()
    scalar_loss(ref).backward()
    for x, y in zip(args['tensors'], refs['tensors']):
        torch.testing.assert_close(x.grad, y.grad)


@pytest.mark.parametrize('dynamic', [None, False, True])
def test_changed_shapes_empty_tensors_and_noncontiguous_inputs(dynamic):
    compiled = wrap(lambda x: (x.sin() * x).sum(-1), dynamic=dynamic)
    for rows in (2, 5, 0, 3):
        x = torch.randn(4, rows).t().detach().requires_grad_()
        ref = x.detach().clone().requires_grad_()
        actual = compiled(x)
        expected = (ref.sin() * ref).sum(-1)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(x.grad, ref.grad)


def test_data_dependent_graph_break_is_not_retried():
    effects = []
    def fn(x):
        effects.append('call')
        if x.sum().item() > 0:
            return x.sin()
        return x.cos()
    compiled = wrap(fn, fullgraph=False)
    for value in (1., -1., 2.):
        x = torch.full((3,), value, requires_grad=True)
        actual = compiled(x)
        expected = x.sin() if value > 0 else x.cos()
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
    assert effects == ['call', 'call', 'call']


def test_fullgraph_rejects_graph_break_instead_of_claiming_native_coverage():
    def fn(x):
        torch._dynamo.graph_break()
        return x * x
    compiled = wrap(fn, fullgraph=True)
    with pytest.raises(torch._dynamo.exc.Unsupported):
        compiled(torch.ones(3))


def test_parameter_identity_tied_weights_checkpoints_and_train_eval():
    model = SharedEmbedding()
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    compiled = wrap(model)
    assert compiled.original is model and compiled.training is False
    assert compiled.embedding is model.embedding
    assert list(compiled.parameters())[0] is optimizer.param_groups[0]['params'][0]
    assert list(compiled.named_parameters()) == list(model.named_parameters())
    assert list(compiled.state_dict()) == list(model.state_dict())
    state = copy.deepcopy(compiled.state_dict())
    compiled.train()
    assert compiled.training and model.training
    compiled.eval()
    assert not compiled.training and not model.training
    compiled.load_state_dict(state)
    assert model.projection.weight is model.embedding.weight


def test_mode_transitions_running_stats_rng_and_inference_mode():
    torch.manual_seed(12)
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Dropout(.3), torch.nn.Linear(4, 2))
    reference = copy.deepcopy(model)
    compiled = wrap(model)
    for training in (True, False, True):
        compiled.train(training)
        reference.train(training)
        x = torch.randn(3, 4)
        torch.manual_seed(29)
        expected = reference(x)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(29)
        actual = compiled(x)
        assert_tree(actual, expected)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert_tree(model.state_dict(), reference.state_dict())
    compiled.eval()
    reference.eval()
    with torch.inference_mode():
        x = torch.randn(3, 4)
        assert_tree(compiled(x), reference(x))


def test_multiple_outstanding_forwards_gradient_accumulation_and_retained_graph():
    model = Residual()
    reference = copy.deepcopy(model)
    compiled = wrap(model, native='auto', min_native_ops=1)
    x = torch.randn(3, 8)
    a, b = compiled(x), compiled(x * 2)
    ea, eb = reference(x), reference(x * 2)
    loss, expected = scalar_loss(a) + scalar_loss(b), scalar_loss(ea) + scalar_loss(eb)
    loss.backward(retain_graph=True)
    expected.backward(retain_graph=True)
    loss.backward()
    expected.backward()
    for p, r in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, r.grad)
    torch.testing.assert_close(a, ea)


def test_output_alias_and_input_mutation_semantics():
    def fn(x):
        x.add_(2)
        return {'view': x[:, :2], 'same': x, 'again': x}
    compiled = wrap(fn, native='auto', min_native_ops=1)
    x = torch.randn(3, 4)
    ref = x.clone()
    result, expected = compiled(x), fn(ref)
    assert_tree(result, expected)
    torch.testing.assert_close(x, ref)
    assert result['same'] is result['again']
    assert result['view'].untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    result['view'].add_(1)
    expected['view'].add_(1)
    torch.testing.assert_close(x, ref)


def test_custom_autograd_function_is_differentiated_by_pytorch():
    class Cubic(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x ** 3
        @staticmethod
        def backward(ctx, g):
            x, = ctx.saved_tensors
            return 3 * x * x * g
    compiled = wrap(Cubic.apply)
    x = torch.randn(4, requires_grad=True)
    compiled(x).sum().backward()
    torch.testing.assert_close(x.grad, 3 * x.detach().square())


def test_registered_custom_operator_with_fake_and_autograd():
    @torch.library.custom_op('ruda_compile_tests::squared', mutates_args=())
    def squared(x: torch.Tensor) -> torch.Tensor:
        return x.square()
    @squared.register_fake
    def fake(x):
        return torch.empty_like(x)
    def setup(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return 2 * x * g
    squared.register_autograd(backward, setup_context=setup)
    compiled = wrap(lambda x: squared(x).sum())
    x = torch.randn(3, requires_grad=True)
    compiled(x).backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())


def test_operator_decompositions_are_applied():
    decompositions = {torch.ops.aten.silu.default: lambda x: x * x.sigmoid()}
    compiled = wrap(torch.nn.SiLU(), decompositions=decompositions)
    x = torch.randn(4, requires_grad=True)
    r = x.detach().clone().requires_grad_()
    actual, expected = compiled(x), torch.nn.functional.silu(r)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, r.grad)
    assert 'aten.silu.default' not in str(compiled.info['graphs'][0]['operators'])


def test_whole_training_callable_including_optimizer_updates():
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.05, momentum=.1)
    def step(x, target):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach()
    compiled = wrap(step, fullgraph=False)
    for _ in range(3):
        x, target = torch.randn(4, 3), torch.randn(4, 2)
        reference_optimizer.zero_grad(set_to_none=True)
        expected = (reference(x) - target).square().mean()
        expected.backward()
        reference_optimizer.step()
        actual = compiled(x, target)
        torch.testing.assert_close(actual, expected.detach())
        assert_tree(model.state_dict(), reference.state_dict())


def test_eager_explicit_mode_and_second_derivatives():
    calls = []
    def fn(x):
        calls.append(1)
        return x ** 3
    compiled = wrap(fn, capture='eager')
    x = torch.randn(4, requires_grad=True)
    g, = torch.autograd.grad(compiled(x).sum(), x, create_graph=True)
    gg, = torch.autograd.grad(g.sum(), x)
    torch.testing.assert_close(gg, 6 * x)
    assert compiled.info['eager_calls'] == 1
    assert compiled.info['graphs'] == []
    assert calls == [1]


def test_aot_second_derivative_is_not_claimed():
    compiled = wrap(lambda x: x ** 3)
    x = torch.randn(4, requires_grad=True)
    g, = torch.autograd.grad(compiled(x).sum(), x, create_graph=True)
    with pytest.raises(RuntimeError, match='double backward'):
        torch.autograd.grad(g.sum(), x)


def test_decorator_form_and_direct_backend():
    @compiler.compile(device_type='cpu', native='off')
    def squared(x):
        return x * x
    x = torch.randn(4, requires_grad=True)
    squared(x).sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())
    backend = compiler.make_backend(device_type='cpu', native='off')
    compiled = torch.compile(lambda x: x.cos(), backend=backend)
    compiled(x).sum().backward()
    assert backend.info['backward_calls'] == 1
    backend.close()


def test_missing_device_is_not_silently_moved():
    with pytest.raises(ValueError, match='Move model and inputs explicitly'):
        compiler.compile(torch.nn.Linear(3, 2))
    compiled = compiler.compile(lambda x: x, capture='eager')
    with pytest.raises(ValueError, match='expected ruda'):
        compiled({'x': [torch.ones(3)]})


def test_exception_side_effects_are_not_retried():
    effects = []
    def bad(x):
        effects.append('entered')
        raise RuntimeError('user failure')
    compiled = wrap(bad, capture='eager')
    with pytest.raises(RuntimeError, match='user failure'):
        compiled(torch.ones(1))
    assert effects == ['entered']


def test_required_native_rejects_unsupported_operator():
    compiled = wrap(lambda x: x.erf(), native='required')
    with pytest.raises(Exception, match='not entirely native'):
        compiled(torch.randn(3, requires_grad=True))


def test_required_native_rejects_missing_extension_before_execution():
    compiled = wrap(lambda x: x + x, native='required')
    with pytest.raises(compiler.NativeCoverageError, match='extension unavailable'):
        compiled(torch.randn(3, requires_grad=True))


def test_info_is_detached_and_close_is_idempotent():
    compiled = wrap(lambda x: x * x)
    compiled(torch.ones(2))
    info = compiled.info
    info['graphs'].clear()
    assert compiled.info['graphs']
    compiled.close()
    compiled.close()
    assert compiled.info['closed']
    with pytest.raises(RuntimeError, match='closed'):
        compiled(torch.ones(2))


@pytest.mark.parametrize('kwargs', [
    {'capture': 'invalid'}, {'native': 'invalid'}, {'cache_size': 0}, {'cache_size': True},
    {'min_native_ops': 0}, {'min_native_ops': 257}, {'fullgraph': 1}, {'dynamic': 'auto'},
    {'capture': 'eager', 'fullgraph': True}, {'capture': 'eager', 'native': 'required'},
    {'decompositions': {'wrong': lambda x: x}}, {'decompositions': []},
])
def test_invalid_options(kwargs):
    with pytest.raises((TypeError, ValueError)):
        wrap(lambda x: x, **kwargs)


def test_global_error_suppression_is_rejected_without_mutating_it():
    with torch._dynamo.config.patch(suppress_errors=True):
        with pytest.raises(ValueError, match='suppress_errors must be False'):
            wrap(lambda x: x * x)
        assert torch._dynamo.config.suppress_errors is True
    compiled = wrap(lambda x: x * x)
    with torch._dynamo.config.patch(suppress_errors=True):
        with pytest.raises(ValueError, match='suppress_errors must be False'):
            compiled(torch.ones(2))
    assert compiled.info['graphs'] == []


def test_module_backward_hooks_survive_graph_breaks():
    model = torch.nn.Linear(3, 2)
    called = []
    model.register_full_backward_hook(lambda *args: called.append('backward'))
    compiled = wrap(model)
    x = torch.randn(2, 3, requires_grad=True)
    compiled(x).sum().backward()
    assert called == ['backward']


def test_frozen_and_unused_parameters_keep_none_gradients():
    class Partial(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(3))
            self.frozen = torch.nn.Parameter(torch.ones(3), requires_grad=False)
            self.unused = torch.nn.Parameter(torch.ones(3))
        def forward(self, x):
            return x * self.w + self.frozen
    model = Partial()
    compiled = wrap(model)
    compiled(torch.ones(3)).sum().backward()
    assert model.frozen.grad is None and model.unused.grad is None
    torch.testing.assert_close(model.w.grad, torch.ones(3))


def test_non_reentrant_activation_checkpoint_preserves_gradients():
    from torch.utils.checkpoint import checkpoint
    def fn(x):
        return checkpoint(lambda t: t.sin() * t, x, use_reentrant=False)
    compiled = wrap(fn)
    x = torch.randn(3, requires_grad=True)
    r = x.detach().clone().requires_grad_()
    actual, expected = compiled(x), fn(r)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, r.grad)


def test_nested_wrapper_checkpoint_roundtrip_preserves_structural_prefix():
    compiled = wrap(torch.nn.Linear(3, 2))
    parent = torch.nn.ModuleDict({'child': compiled})
    state = copy.deepcopy(parent.state_dict())
    assert list(state) == ['child._original.weight', 'child._original.bias']
    with torch.no_grad():
        compiled.weight.zero_()
    parent.load_state_dict(state)
    torch.testing.assert_close(compiled.weight, state['child._original.weight'])
