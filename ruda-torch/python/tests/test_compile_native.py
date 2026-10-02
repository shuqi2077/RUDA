"""Native-region ownership/partition tests with an explicitly injected CPU simulator.

These run the production partitioner, staging and AOT integration, but NOT PTX,
Rust, C++ dispatch, native stream synchronization or real GPU graph capture.
"""
import copy
import importlib
from pathlib import Path
import sys
import types

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

NAME = 'ruda_native_partition_test'
package = types.ModuleType(NAME)
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'ruda_torch')]
package._graph_available = False
sys.modules[NAME] = package
compiler = importlib.import_module(NAME + '.compiler')
native = importlib.import_module(NAME + '._compile_native')
spec = importlib.import_module(NAME + '._graph_spec')


class Simulator:
    def __init__(self, inputs, nodes, outputs):
        self.inputs = inputs
        self.nodes = nodes
        self.outputs = outputs
        self.workspace = {}
        self.closed = False
        self.calls = 0
        self.fail = False

    def replay(self):
        assert not self.closed
        self.calls += 1
        if self.fail:
            raise RuntimeError('simulated native dispatch failure')
        values = dict(self.inputs)
        for node in self.nodes:
            x = values[node.left]
            if node.kind == 'add':
                y = torch.add(x, values[node.right], alpha=node.scalar)
            elif node.kind == 'mul':
                y = x * values[node.right]
            elif node.kind == 'copy':
                y = x.clone()
            elif node.kind == 'silu':
                y = torch.nn.functional.silu(x)
            elif node.kind in spec.UNARY_CODES:
                y = getattr(torch.ops.aten, node.kind).default(x)
            elif node.kind in ('mm','bmm'):
                y = getattr(torch, node.kind)(x, values[node.right])
            elif node.kind == 'div': y = x / values[node.right]
            elif node.kind in ('add_scalar','mul_scalar','div_scalar'):
                y = {'add_scalar':torch.add,'mul_scalar':torch.mul,'div_scalar':torch.div}[node.kind](x,node.scalar)
            elif node.kind in ('silu_backward','sigmoid_backward','tanh_backward'):
                y = getattr(torch.ops.aten, node.kind).default(x, values[node.right])
            elif node.kind in ('softmax','log_softmax'):
                y = getattr(torch, node.kind)(x, dim=int(node.scalar))
            elif node.kind in ('softmax_backward','log_softmax_backward'):
                target = torch.ops.aten._softmax_backward_data if node.kind == 'softmax_backward' else torch.ops.aten._log_softmax_backward_data
                y = target.default(x, values[node.right], int(node.scalar), x.dtype)
            elif node.kind in ('sum_keepdim','mean_keepdim'):
                dims = tuple(i for i in range(x.ndim) if int(node.scalar) & (1<<i))
                y = getattr(x,'sum' if node.kind == 'sum_keepdim' else 'mean')(dims,keepdim=True)
            else:
                raise AssertionError(node.kind)
            if node.output not in self.workspace:
                self.workspace[node.output] = torch.empty_like(y)
            self.workspace[node.output].copy_(y)
            values[node.output] = self.workspace[node.output]
        return {name: values[name] for name in self.outputs}

    def close(self):
        self.closed = True


@pytest.fixture
def simulated(monkeypatch):
    torch._dynamo.reset()
    graphs = []
    stream = [0]
    def build(*args):
        graph = Simulator(*args)
        graphs.append(graph)
        return graph
    def cpu_spec(t):
        if not isinstance(t, torch.Tensor) or t.device.type != 'cpu':
            raise ValueError('explicit CPU simulator only')
        if t.layout != torch.strided:
            raise ValueError('native regions require dense strided tensors')
        return spec.TensorSpec(tuple(t.shape), str(t.dtype).removeprefix('torch.'))
    monkeypatch.setattr(native, '_native_available', lambda: True)
    monkeypatch.setattr(native, '_stream_id', lambda: stream[0])
    monkeypatch.setattr(native, '_build_graph', build)
    monkeypatch.setattr(native, '_input_spec', cpu_spec)
    yield graphs, stream
    torch._dynamo.reset()


def make_call(fn, inputs, **options):
    backend = compiler.make_backend(device_type='cpu', min_native_ops=1, **options)
    gm = make_fx(fn)(*inputs)
    return backend, backend._compiler('forward', gm, inputs)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_aot_native_regions_compute_automatic_gradients(simulated, dtype):
    def fn(x, u):
        v = torch.nn.functional.silu(x + u)
        return v * u + x
    compiled = compiler.compile(fn, device_type='cpu', min_native_ops=1)
    torch.manual_seed(483)
    x = torch.randn(3, 7, dtype=dtype, requires_grad=True)
    u = torch.randn(3, 7, dtype=dtype, requires_grad=True)
    rx, ru = x.detach().clone().requires_grad_(), u.detach().clone().requires_grad_()
    y, expected = compiled(x, u), fn(rx, ru)
    y.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(x.grad, rx.grad)
    torch.testing.assert_close(u.grad, ru.grad)
    assert compiled.info['native_replays'] > 0
    assert any(g['phase'] == 'backward' and g['native_regions'] for g in compiled.info['graphs'])
    compiled.close()
    assert all(g.closed for g in simulated[0])


def test_shared_tied_parameters_and_two_outstanding_forwards(simulated):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(3, 7))
        def forward(self, x):
            z = x * self.w
            return z * z + self.w
    model = Model()
    reference = copy.deepcopy(model)
    compiled = compiler.compile(model, device_type='cpu', min_native_ops=1)
    x = torch.randn(3, 7)
    a, b = compiled(x), compiled(2 * x)
    ea, eb = reference(x), reference(2 * x)
    torch.testing.assert_close(a, ea)
    assert a.data_ptr() != b.data_ptr()
    # A later replay must not overwrite saved AOT activations or returned results.
    (a.square().sum() + b.square().sum()).backward()
    (ea.square().sum() + eb.square().sum()).backward()
    torch.testing.assert_close(model.w.grad, reference.w.grad)
    torch.testing.assert_close(a, ea)


def test_cache_uses_shapes_and_streams_not_input_addresses(simulated):
    graphs, stream = simulated
    fn = lambda x, y: (x + y) * y
    x, y = torch.randn(2, 3), torch.randn(2, 3)
    backend, call = make_call(fn, [x, y], cache_size=2)
    a = call([x, y])
    for _ in range(3):
        x2, y2 = torch.randn(2, 3), torch.randn(2, 3)
        torch.testing.assert_close(call([x2, y2]), fn(x2, y2))
    assert len(graphs) == 1 and graphs[0].calls == 4
    torch.testing.assert_close(a, fn(x, y))
    stream[0] = 1
    call([x, y])
    assert len(graphs) == 2
    stream[0] = 0
    call([torch.randn(5, 3), torch.randn(5, 3)])
    assert len(graphs) == 3 and graphs[0].closed
    assert backend.info['native_cache_hits'] == 3
    assert backend.info['native_cache_evictions'] == 1
    backend.close()
    assert all(g.closed for g in graphs)


def test_cache_separates_inference_tensor_lifetime(simulated):
    x = torch.randn(2, 3)
    backend, call = make_call(lambda x: x * x, [x])
    with torch.inference_mode():
        call([x])
    call([x])
    assert len(simulated[0]) == 2
    backend.close()


@pytest.mark.parametrize('case', ['broadcast', 'noncontiguous', 'empty', 'rank0', 'integer', 'rank9'])
def test_runtime_metadata_guard_uses_reference_before_dispatch(simulated, case):
    if case == 'broadcast':
        args = [torch.randn(2, 3), torch.randn(1, 3)]
    elif case == 'noncontiguous':
        args = [torch.randn(3, 2).t(), torch.randn(3, 2).t()]
    elif case == 'empty':
        args = [torch.empty(0, 3), torch.empty(0, 3)]
    elif case == 'rank0':
        args = [torch.randn(()), torch.randn(())]
    elif case == 'integer':
        args = [torch.ones(2, 3, dtype=torch.int64)] * 2
    else:
        args = [torch.ones([1] * 9)] * 2
    fn = lambda x, y: (x + y) * y
    backend, call = make_call(fn, args)
    torch.testing.assert_close(call(args), fn(*args))
    assert not simulated[0]
    assert backend.info['region_reference_runs'] == (0 if case == 'noncontiguous' else 1)
    if case == 'noncontiguous':
        assert not backend.info['graphs'][0]['native_regions']
    else:
        assert backend.info['reference_reasons']


def test_required_native_runtime_guard_rejects_broadcast(simulated):
    args = [torch.randn(2, 3), torch.randn(1, 3)]
    _, call = make_call(lambda x, y: x + y, args, native='required')
    with pytest.raises(native.NativeCoverageError, match='broadcasting'):
        call(args)
    assert not simulated[0]


def test_failed_native_dispatch_is_not_retried(simulated):
    x = torch.randn(2, 3)
    backend, call = make_call(lambda x: x * x, [x])
    call([x])
    graph = simulated[0][0]
    graph.fail = True
    with pytest.raises(compiler.GraphExecutionError, match='No retry') as caught:
        call([x])
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert graph.calls == 2
    assert backend.info['region_reference_runs'] == 0
    assert backend.info['execution_errors'] == 1


def test_failed_native_build_is_not_treated_as_capability_fallback(simulated, monkeypatch):
    x = torch.ones(2, 3)
    def fail(*_):
        raise RuntimeError('allocation failed')
    monkeypatch.setattr(native, '_build_graph', fail)
    backend, call = make_call(lambda x: x * x, [x])
    with pytest.raises(compiler.GraphExecutionError, match='allocation failed'):
        call([x])
    assert backend.info['region_reference_runs'] == 0


def test_trace_has_no_native_allocation_or_data_access(simulated):
    x = torch.randn(2, 3)
    backend, call = make_call(lambda x: x * x, [x])
    assert not simulated[0]
    call([x])
    assert len(simulated[0]) == 1


def test_partition_preserves_views_mutations_and_multioutput(simulated):
    def fn(x):
        a = x + x
        b = a * a
        v = b.view(-1)
        v.add_(2)
        c = a * a
        return b, v, c
    x = torch.randn(2, 3)
    backend, call = make_call(fn, [x])
    got, expected = call([x]), fn(x)
    for g, e in zip(got, expected):
        torch.testing.assert_close(g, e)
    assert got[0].untyped_storage().data_ptr() == got[1].untyped_storage().data_ptr()
    report = backend.info['graphs'][0]
    assert 'aten.add_.Tensor' in report['device_dispatch_ops']
    assert 'aten.view.default' in report['device_dispatch_ops']
    assert report['native_regions'] == 2


def test_long_regions_are_split_at_native_node_limit(simulated):
    def fn(x):
        for _ in range(260):
            x = x + x
        return x
    x = torch.zeros(2)
    backend, call = make_call(fn, [x])
    torch.testing.assert_close(call([x]), x)
    assert backend.info['graphs'][0]['native_regions'] == 2
    assert all(len(g.nodes) <= 256 for g in simulated[0])


def test_default_device_guard_never_accepts_cpu_without_test_injection():
    with pytest.raises(ValueError, match='no implicit device transfer'):
        native._input_spec(torch.ones(2))


def test_close_prevents_lazy_backward_after_forward(simulated):
    compiled = compiler.compile(lambda x: x * x, device_type='cpu', min_native_ops=1)
    x = torch.randn(2, requires_grad=True)
    y = compiled(x)
    compiled.close()
    with pytest.raises(RuntimeError, match='closed'):
        y.sum().backward()


def test_failed_close_preserves_handles_for_explicit_retry(simulated, monkeypatch):
    x = torch.ones(2, 3)
    backend, call = make_call(lambda x: x * x, [x])
    call([x])
    graph = simulated[0][0]
    original = graph.close
    attempts = []
    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError('simulated synchronization failure')
        original()
    monkeypatch.setattr(graph, 'close', flaky)
    with pytest.raises(RuntimeError, match='release'):
        backend.close()
    assert not graph.closed and backend.info['closed']
    backend.close()
    assert graph.closed and len(attempts) == 2


def test_lru_close_failure_does_not_drop_old_handle(simulated, monkeypatch):
    graphs, stream = simulated
    x = torch.ones(2, 3)
    backend, call = make_call(lambda x: x * x, [x], cache_size=1)
    call([x])
    original = graphs[0].close
    def fail():
        raise RuntimeError('simulated close failure')
    monkeypatch.setattr(graphs[0], 'close', fail)
    stream[0] = 1
    with pytest.raises(compiler.GraphExecutionError, match='close failure'):
        call([x])
    assert len(graphs) == 1 and not graphs[0].closed
    monkeypatch.setattr(graphs[0], 'close', original)
    backend.close()
    assert graphs[0].closed
