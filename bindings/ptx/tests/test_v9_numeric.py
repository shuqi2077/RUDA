"""Execute NEW emitted PTX in the test interpreter and, separately, on a GPU.

A debug pass does not validate NVIDIA assembly, timing, occupancy, or races.
"""
import pytest
import torch
import torch.nn.functional as F
from ruda_ptx import Executor, TensorSpec, TopKSession, compile_exported
from ruda_ptx.decode_linear import linear_decode, gated_linear
from test_v8_numeric import ptx_runtime, buffers, spec, tol


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tile", [2, 4])
@pytest.mark.parametrize("m,k,n,bias", [(1, 1, 1, False), (2, 17, 7, True), (1, 33, 17, False), (4, 65, 9, True)])
def test_tiled_decode_ptx(ptx_runtime, dtype, tile, m, k, n, bias):
    torch.manual_seed(61)
    x, w, b = torch.randn(m, k, dtype=dtype)*0.2, torch.randn(n, k, dtype=dtype)*0.2, torch.randn(n, dtype=dtype)
    with buffers(ptx_runtime) as (alloc, upload, read):
        kernel = linear_decode("tiled", spec(x), spec(w), bias=bias, outputs_per_warp=tile)
        args = tuple(upload(t).buffer for t in ((x, w, b) if bias else (x, w)))
        outspec = TensorSpec((m, n), str(dtype).removeprefix("torch."))
        output = alloc(outspec.nbytes)
        ptx_runtime.launch(ptx_runtime.load(kernel), kernel, args+(output,))
        expected = F.linear(x.float(), w.float(), b.float() if bias else None).to(dtype)
        torch.testing.assert_close(read(output, outspec), expected, rtol=tol(dtype), atol=tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gb,ub", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (2, 33, 7)])
def test_gated_projection_ptx(ptx_runtime, dtype, gb, ub, m, k, n):
    torch.manual_seed(63)
    x = torch.randn(m, k, dtype=dtype)*0.4
    wg, wu = torch.randn(n, k, dtype=dtype)*0.3, torch.randn(n, k, dtype=dtype)*0.3
    bg, bu = torch.randn(n, dtype=dtype), torch.randn(n, dtype=dtype)
    with buffers(ptx_runtime) as (alloc, upload, read):
        kernel = gated_linear("gate", spec(x), spec(wg), gate_bias=gb, up_bias=ub)
        args = (x, wg, wu)+((bg,) if gb else ())+((bu,) if ub else ())
        params = tuple(upload(t).buffer for t in args)
        outspec = TensorSpec((m, n), str(dtype).removeprefix("torch."))
        output = alloc(outspec.nbytes)
        ptx_runtime.launch(ptx_runtime.load(kernel), kernel, params+(output,))
        gate = F.linear(x.float(), wg.float(), bg.float() if gb else None).to(dtype)
        up = F.linear(x.float(), wu.float(), bu.float() if ub else None).to(dtype)
        expected = (F.silu(gate.float()).to(dtype).float()*up.float()).to(dtype)
        torch.testing.assert_close(read(output, outspec), expected, rtol=tol(dtype), atol=tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width,k,parts", [(1, 1, 8), (8, 8, 16), (17, 4, 3), (129, 8, 2), (1031, 1, 4), (1031, 8, 3)])
def test_partitioned_topk_ptx(ptx_runtime, dtype, width, k, parts):
    rt = ptx_runtime
    torch.manual_seed(65)
    # Ties across partition boundaries must keep the original index ordering.
    x = torch.randint(-3, 4, (2, width)).to(dtype)
    with buffers(rt) as (_, upload, read), TopKSession(rt, spec(x), k, partitions=parts) as session:
        dx = upload(x)
        first = session.run(dx)
        again = session.run(dx, synchronize=True)
        assert first is again
        indices = torch.frombuffer(bytearray(rt.read(again.indices, session.program.output_nbytes)), dtype=torch.int32).reshape(2, k)
        values = read(again.values.buffer, again.values.spec)
        expected = torch.argsort(x, dim=-1, descending=True, stable=True)[..., :k]
        assert torch.equal(indices.long(), expected)
        torch.testing.assert_close(values, x.gather(-1, expected).float(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_exported_gated_mlp_numeric(ptx_runtime, dtype):
    class Gated(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(17, 33, dtype=dtype)
            self.up = torch.nn.Linear(17, 33, bias=False, dtype=dtype)
        def forward(self, x):
            return F.silu(self.gate(x))*self.up(x)
    torch.manual_seed(70)
    model = Gated().eval()
    x = torch.randn(2, 17, dtype=dtype)*0.25
    ep = torch.export.export(model, (x,))
    p = compile_exported(ep)
    old = compile_exported(ep, fuse_gated_decode=False)
    assert len(p.steps) == 1 and len(old.steps) == 3
    assert p.report()["workspace_bytes"] < old.report()["workspace_bytes"]
    with Executor(p, ptx_runtime) as new, Executor(old, ptx_runtime) as prior, torch.inference_mode():
        # Compare to BOTH the retained unfused PTX path and the model.
        result = new(x)
        torch.testing.assert_close(result, prior(x), rtol=tol(dtype), atol=tol(dtype))
        torch.testing.assert_close(result, model(x), rtol=tol(dtype), atol=tol(dtype))
