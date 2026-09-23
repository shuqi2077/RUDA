"""Vector PTX: debug interpretation and physical GPU are separate test cases."""
import pytest
import torch
import torch.nn.functional as F
from ruda_ptx import Executor, TensorSpec, compile_exported
from ruda_ptx.vectorized import elementwise4
from test_v8_numeric import ptx_runtime, buffers, spec
from ptx_debug import DebugRuntime


@pytest.mark.parametrize("op", ["add", "mul", "silu", "silu_mul"])
@pytest.mark.parametrize("n", [1, 3, 4, 5, 255, 1024, 1025])
def test_vector_ptx(ptx_runtime, op, n):
    torch.manual_seed(n)
    x, y = torch.randn(n), torch.randn(n)
    expected = {"add": lambda: x+y, "mul": lambda: x*y,
                "silu": lambda: F.silu(x), "silu_mul": lambda: F.silu(x)*y}[op]()
    with buffers(ptx_runtime) as (alloc, upload, read):
        kernel = elementwise4("ruda_vector", op, spec(x))
        inputs = (upload(x).buffer,) if op == "silu" else (upload(x).buffer, upload(y).buffer)
        out = alloc(x.numel()*4)
        ptx_runtime.launch(ptx_runtime.load(kernel), kernel, inputs+(out,))
        torch.testing.assert_close(read(out, spec(x)), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("op", ["add", "mul", "silu", "silu_mul"])
@pytest.mark.parametrize("n", [3, 12, 17])
def test_vector_alignment_scalar_path(op, n):
    rt = DebugRuntime()
    rt.next += 4
    x, y = torch.arange(n, dtype=torch.float32)/10, torch.ones(n)
    with buffers(rt) as (alloc, upload, read):
        kernel = elementwise4("misaligned", op, spec(x))
        inputs = (upload(x).buffer,) if op == "silu" else (upload(x).buffer, upload(y).buffer)
        assert inputs[0].handle % 16 == 4
        out = alloc(4*n)
        rt.launch(rt.load(kernel), kernel, inputs+(out,))
        expected = {"add": x+y, "mul": x*y, "silu": F.silu(x), "silu_mul": F.silu(x)*y}[op]
        torch.testing.assert_close(read(out, spec(x)), expected)


def test_vector_graph_is_opt_in_and_same_output():
    class M(torch.nn.Module):
        def forward(self, x, y): return F.silu(x+y)*y
    x, y = torch.randn(1, 1025), torch.randn(1, 1025)
    ep = torch.export.export(M(), (x, y))
    old, new = compile_exported(ep), compile_exported(ep, vectorized_elementwise=True)
    assert "v4.f32" not in old.steps[0].kernel.ptx
    assert "v4.f32" in new.steps[0].kernel.ptx
    assert old.workspace() == new.workspace()
    rt = DebugRuntime()
    with Executor(old, rt) as a, Executor(new, rt) as b, torch.inference_mode():
        torch.testing.assert_close(a(x, y), b(x, y), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_vector_standalone_rejects_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="float32"):
        elementwise4("bad", "add", TensorSpec((4,), dtype))
