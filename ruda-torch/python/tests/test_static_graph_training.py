"""Autograd/VJP regression tests using an explicit CPU forward simulator.

These execute PyTorch autograd, but do NOT validate native Rust/PTX dispatch.
Production StaticGraph continues to accept only ruda:0 device tensors.
"""
import importlib
from pathlib import Path
import sys
import types
import pytest
import torch

name = 'ruda_training_host_test'
package = types.ModuleType(name)
package.__path__ = [str(Path(__file__).resolve().parents[1]/'ruda_torch')]
sys.modules[name] = package
spec = importlib.import_module(name+'._graph_spec')
opt = importlib.import_module(name+'._graph_opt')
ad = importlib.import_module(name+'._graph_autograd')
Op = spec.GraphOp


class ForwardSimulator:
    def __init__(self, inputs, nodes, outputs=None, *, optimize=False, reuse=False):
        self.inputs = tuple(inputs.values())
        metadata = {n: spec.TensorSpec(tuple(x.shape), str(x.dtype).removeprefix('torch.'))
                    for n, x in inputs.items()}
        plan = opt.prepare_plan(metadata, nodes, outputs, optimize=optimize, reuse_workspace=reuse)
        self._layout = plan.layout
        self._tensors = list(self.inputs)
        for i in range(plan.layout.inputs, len(plan.layout.specs)):
            s = plan.layout.specs[i]
            root = plan.storage_roots[i]
            self._tensors.append(torch.empty(s.shape, dtype=getattr(torch, s.dtype))
                                 if root == i else self._tensors[root])
        self._native = self
        self.closed = False
        self.calls = 0

    def _check(self):
        if self.closed:
            raise RuntimeError('closed')

    def _check_bindings(self):
        pass

    def run(self, eager):
        self.calls += 1
        values = ad.forward_values(self._layout, self.inputs)
        for i in range(self._layout.inputs, len(values)):
            self._tensors[i].copy_(values[i])

    def replay(self):
        return ad.StaticGraphFunction.apply(self, False, *self.inputs)


def make_inputs(dtype, rank=2):
    torch.manual_seed(638)
    shape = (3, 7) if rank == 2 else (7,)
    return {'x': torch.randn(shape, dtype=dtype, requires_grad=True),
            'u': torch.randn(shape, dtype=dtype, requires_grad=True),
            'w': torch.randn(7, dtype=dtype, requires_grad=True)}


def compare(inputs, nodes, outputs=None, *, optimize=False):
    graph = ForwardSimulator(inputs, nodes, outputs, optimize=optimize, reuse=True)
    reference = {n: x.detach().clone().requires_grad_(x.requires_grad) for n, x in inputs.items()}
    actual = graph.replay()
    vals = ad.forward_values(graph._layout, reference.values())
    expected = tuple(vals[i] for i in graph._layout.output_indices)
    torch.manual_seed(282)
    seeds = tuple(torch.randn_like(y) for y in actual)
    got = torch.autograd.grad(actual, tuple(inputs.values()), seeds, allow_unused=True)
    want = torch.autograd.grad(expected, tuple(reference.values()), seeds, allow_unused=True)
    dtype = next(iter(inputs.values())).dtype
    tol = {torch.float32: 2e-6, torch.float16: 4e-3, torch.bfloat16: 4e-2}[dtype]
    for y, ref in zip(actual, expected):
        torch.testing.assert_close(y, ref, rtol=0, atol=0)
    for g, ref in zip(got, want):
        if ref is None:
            assert g is None
        else:
            torch.testing.assert_close(g, ref, rtol=tol, atol=tol)
    assert graph.calls == 1


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('kind', ['copy', 'add', 'mul', 'silu', 'silu_mul', 'rms_norm', 'rms_no_weight'])
@pytest.mark.parametrize('rank', [1, 2])
def test_every_operator_gradient(dtype, kind, rank):
    node = {'copy': Op.copy('y','x'), 'add': Op.add('y','x','u',alpha=-.375),
            'mul': Op.mul('y','x','u'), 'silu': Op.silu('y','x'),
            'silu_mul': Op.silu_mul('y','x','u'),
            'rms_norm': Op.rms_norm('y','x','w',eps=.003),
            'rms_no_weight': Op.rms_norm('y','x',eps=.003)}[kind]
    compare(make_inputs(dtype, rank), [node])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('optimize', [False, True])
def test_fanout_multioutput_duplicate_edges_and_fusion(dtype, optimize):
    nodes = [Op.silu('a','x'), Op.mul('b','a','u'), Op.add('c','b','b',alpha=.5),
             Op.rms_norm('d','c','w'), Op.add('y','d','x')]
    compare(make_inputs(dtype), nodes, ['b','y'], optimize=optimize)


def test_two_outstanding_forwards_own_outputs_and_gradients():
    x = torch.tensor([.25, -.75], requires_grad=True)
    graph = ForwardSimulator({'x': x}, [Op.mul('y','x','x')])
    a, = graph.replay()
    b, = graph.replay()
    assert a.data_ptr() != b.data_ptr()
    assert a.data_ptr() != graph._tensors[-1].data_ptr()
    graph._tensors[-1].fill_(999)
    (a.sum()+b.sum()).backward()
    torch.testing.assert_close(x.grad, 4*x.detach())


def test_inplace_mutation_is_rejected():
    x = torch.ones(3, requires_grad=True)
    graph = ForwardSimulator({'x':x}, [Op.mul('y','x','x')])
    y, = graph.replay()
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        y.sum().backward()


def test_close_after_forward_does_not_break_backward():
    x = torch.ones(3, requires_grad=True)
    graph = ForwardSimulator({'x':x}, [Op.mul('y','x','x')])
    y, = graph.replay()
    graph.closed = True
    graph._tensors = []
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2))


def test_unused_outputs_and_non_grad_inputs():
    x = torch.ones(3, requires_grad=True)
    u = torch.full((3,), 2.)
    graph = ForwardSimulator({'x':x,'u':u},
        [Op.mul('a','x','u'), Op.silu('b','x')], ['a','b'])
    a, _ = graph.replay()
    a.sum().backward()
    torch.testing.assert_close(x.grad, u)
    assert u.grad is None


def test_repeated_optimizer_steps_reduce_loss():
    x = torch.tensor([2., -3.], requires_grad=True)
    graph = ForwardSimulator({'x':x}, [Op.mul('y','x','x')])
    optimizer = torch.optim.SGD([x], lr=.1)
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        y, = graph.replay()
        loss = y.sum()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]*.1


def test_double_backward_is_not_silently_supported():
    x = torch.ones(3, requires_grad=True)
    graph = ForwardSimulator({'x':x}, [Op.mul('y','x','x')])
    y, = graph.replay()
    grad, = torch.autograd.grad(y.sum(), x, create_graph=True)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(grad.sum(), x)
