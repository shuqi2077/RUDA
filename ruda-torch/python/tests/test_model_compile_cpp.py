"""Real C++ device hooks + FakeTensor/AOT tracing; no Rust or GPU execution.

Run in a fresh process with RUDA_CPP_TEST_LIBRARY pointing at the compiled C++
bridge. The imported fixture uses explicit host-only ABI callbacks.
"""
import importlib
from pathlib import Path
import sys
import types

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from test_v15_cpp import bridge, _KEEP_ALIVE


@pytest.fixture(scope='module')
def frontend(bridge):
    cpp, state, _ = bridge
    name = 'ruda_compile_cpp_package'
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'ruda_torch')]
    package._C = cpp
    package._graph_available = False
    package.is_available = lambda: True
    sys.modules[name] = package
    # Same hooks exported by the production package after its ABI initialization.
    torch.ruda.is_initialized = lambda: True
    torch.ruda._lazy_init = lambda: None
    torch.ruda.manual_seed_all = lambda _: None
    compiler = importlib.import_module(name + '.compiler')
    return compiler, state


def test_fake_ruda_forward_capture_never_allocates_native_storage(frontend):
    compiler, state = frontend
    before = len(state.allocations)
    torch._dynamo.reset()
    mode = FakeTensorMode(allow_fallback_kernels=False)
    with mode:
        x = FakeTensor(mode, torch.empty(2, 3, device='meta'), torch.device('ruda'))
        gm = torch.fx.symbolic_trace(lambda x: ((x + x) * x,))
        fn = compiler.make_backend(native='off')
        call = fn(gm, [x])
        y, = call(x)
        assert y.device == torch.device('ruda:0') and y.shape == x.shape
        assert fn.info['inference_calls'] == 1
        fn.close()
    assert len(state.allocations) == before


def test_fake_ruda_backward_is_captured_with_real_device_guard(frontend):
    compiler, state = frontend
    before = len(state.allocations)
    torch._dynamo.reset()
    mode = FakeTensorMode(allow_fallback_kernels=False)
    with mode:
        x = FakeTensor(mode, torch.empty(2, 3, device='meta', requires_grad=True), torch.device('ruda'))
        gm = torch.fx.symbolic_trace(lambda x: ((x + x) * x,))
        fn = compiler.make_backend(native='off')
        call = fn(gm, [x])
        y, = call(x)
        grad, = torch.autograd.grad(y.sum(), x)
        assert grad.device == x.device and grad.shape == x.shape
        assert fn.info['forward_calls'] == 1 and fn.info['backward_calls'] == 1
        fn.close()
    assert len(state.allocations) == before


def test_dynamo_real_ruda_tensor_captures_before_missing_kernel_error(frontend):
    compiler, state = frontend
    torch._dynamo.reset()
    x = torch.empty(2, 3, device='ruda', requires_grad=True)
    before = len(state.allocations)
    fn = compiler.compile(lambda x: x + x, native='off', fullgraph=True)
    # These ABI fixtures intentionally do not register numerical add kernels.
    # A clean failure AFTER capture validates real device tracing, not numerics.
    with pytest.raises(compiler.GraphExecutionError, match='No retry'):
        fn(x)
    assert fn.info['forward_calls'] == 1
    assert fn.info['graphs'][0]['operators'][0]['target'] == 'aten.add.Tensor'
    assert len(state.allocations) == before
    fn.close()


def test_compiler_device_context_is_idempotent_and_rejects_other_device(frontend):
    compiler, _ = frontend
    device = importlib.import_module(compiler.__package__ + '._compile_device')
    device.register_device_interface()
    device.register_device_interface()
    from torch._dynamo.device_interface import get_interface_for_device
    interface = get_interface_for_device('ruda')
    with interface.device('ruda:0'):
        assert interface.current_device() == 0
    with pytest.raises(ValueError, match='only ruda:0'):
        with interface.device(1):
            pass
    with pytest.raises(ValueError, match='expected a ruda device'):
        with interface.device('cpu'):
            pass


def test_static_graph_model_entry_returns_callable_not_fixed_address_graph(frontend):
    compiler, _ = frontend
    module = importlib.import_module(compiler.__package__ + '._graph')
    # Explicit CPU eager mode exercises API routing, without a native tensor call.
    compiled = module.StaticGraph.from_model(lambda x: x * x, device_type='cpu', capture='eager')
    assert isinstance(compiled, compiler.CompiledFunction)
    x = torch.randn(3)
    torch.testing.assert_close(compiled(x), x * x)
    compiled.close()
