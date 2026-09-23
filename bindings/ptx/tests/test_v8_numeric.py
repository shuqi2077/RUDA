"""Same numerical cases for emitted-PTX debug interpretation and actual driver.

A debug pass is NOT a GPU pass. GPU absence is a skip unless RUDA_REQUIRE_GPU=1.
"""
from contextlib import contextmanager
import os
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from ruda_ptx import TensorSpec, DeviceTensor, Executor, compile_exported, StaticKVCache
from ruda_ptx.attention import attention_program
from ruda_ptx.reductions import rms_norm, softmax
from ruda_ptx.rotary import rope
from ruda_ptx.decode_linear import linear_decode
from ruda_ptx.selection import topk
from ptx_debug import DebugRuntime


@pytest.fixture(scope="module", params=[pytest.param("debug", id="ptx-debug"),
                                         pytest.param("gpu", id="gpu", marks=pytest.mark.gpu)])
def ptx_runtime(request):
    if request.param == "debug":
        yield DebugRuntime()
        return
    from ruda_ptx.nvidia_driver import NvidiaDriverRuntime, DriverError
    try:
        runtime = NvidiaDriverRuntime()
    except DriverError as exc:
        if os.environ.get("RUDA_REQUIRE_GPU") == "1":
            pytest.fail(f"Actual GPU required: {exc}")
        pytest.skip(str(exc))
    yield runtime
    runtime.close()


def spec(x):
    return TensorSpec(tuple(x.shape), str(x.dtype).removeprefix("torch."))


def tol(dtype):
    return 8e-4 if dtype == torch.float32 else (6e-3 if dtype == torch.float16 else 4e-2)


@contextmanager
def buffers(runtime):
    owned = []
    def alloc(n):
        b = runtime.allocate(n)
        owned.append(b)
        return b
    def upload(x):
        if x.dtype == torch.bfloat16 and getattr(runtime, "sm", 80) < 80:
            pytest.skip("BF16 variants require sm_80")
        s = spec(x)
        b = alloc(s.nbytes)
        runtime.write(b, Executor._tensor_bytes(x.contiguous()))
        return DeviceTensor(b, s)
    def read(b, s):
        return torch.frombuffer(bytearray(runtime.read(b, s.nbytes)), dtype=getattr(torch, s.dtype)).reshape(s.shape)
    try:
        yield alloc, upload, read
    finally:
        runtime.synchronize()
        for b in reversed(owned):
            runtime.free(b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width,residual", [(1, False), (17, True), (129, False), (257, True), (1031, False)])
def test_norm_ptx(ptx_runtime, dtype, width, residual):
    rt = ptx_runtime
    torch.manual_seed(17)
    x, r = torch.randn(2, width, dtype=dtype)*0.25, torch.randn(2, width, dtype=dtype)*0.1
    w = torch.randn(width, dtype=dtype)
    with buffers(rt) as (alloc, upload, read):
        dx, dw = upload(x), upload(w)
        inputs = (dx.buffer, upload(r).buffer, dw.buffer) if residual else (dx.buffer, dw.buffer)
        kernel = rms_norm("test_norm", spec(x), 1e-6, residual=residual)
        output = alloc(spec(x).nbytes)
        rt.launch(rt.load(kernel), kernel, inputs + (output,))
        source = (x + r).to(dtype) if residual else x
        expected = (source.float() / torch.sqrt(source.float().square().mean(-1, keepdim=True)+1e-6) * w.float()).to(dtype)
        torch.testing.assert_close(read(output, spec(x)), expected, rtol=tol(dtype), atol=tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 17, 257])
def test_softmax_ptx(ptx_runtime, dtype, width):
    torch.manual_seed(19)
    x = torch.randn(2, width, dtype=dtype)*3
    with buffers(ptx_runtime) as (alloc, upload, read):
        dx = upload(x)
        kernel = softmax("test_softmax", spec(x))
        output = alloc(spec(x).nbytes)
        ptx_runtime.launch(ptx_runtime.load(kernel), kernel, (dx.buffer, output))
        torch.testing.assert_close(read(output, spec(x)), x.softmax(-1), rtol=tol(dtype), atol=tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("interleaved", [False, True])
def test_rope_ptx(ptx_runtime, dtype, interleaved):
    torch.manual_seed(22)
    x = torch.randn(1, 2, 3, 19, dtype=dtype)
    angles = torch.randn(3, 8)
    cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
    with buffers(ptx_runtime) as (alloc, upload, read):
        kernel = rope("test_rope", spec(x), 16, interleaved=interleaved)
        args = tuple(upload(v).buffer for v in (x, cos, sin))
        out = alloc(spec(x).nbytes)
        ptx_runtime.launch(ptx_runtime.load(kernel), kernel, args + (out,))
        expected = x.float().clone()
        a, b = (x[..., :16:2], x[..., 1:16:2]) if interleaved else (x[..., :8], x[..., 8:16])
        left = a.float()*cos.float()-b.float()*sin.float()
        right = b.float()*cos.float()+a.float()*sin.float()
        if interleaved:
            expected[..., :16:2], expected[..., 1:16:2] = left, right
        else:
            expected[..., :8], expected[..., 8:16] = left, right
        torch.testing.assert_close(read(out, spec(x)), expected.to(dtype), rtol=tol(dtype), atol=tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,bias", [((1, 17, 5), False), ((2, 33, 7), True)])
def test_decode_linear_ptx(ptx_runtime, dtype, shape, bias):
    torch.manual_seed(31)
    m, k, n = shape
    x, w, b = torch.randn(m, k, dtype=dtype)*0.3, torch.randn(n, k, dtype=dtype)*0.3, torch.randn(n, dtype=dtype)
    with buffers(ptx_runtime) as (alloc, upload, read):
        kernel = linear_decode("test_linear", spec(x), spec(w), bias=bias)
        args = tuple(upload(t).buffer for t in ((x, w, b) if bias else (x, w)))
        outspec = TensorSpec((m, n), str(dtype).removeprefix("torch."))
        out = alloc(outspec.nbytes)
        ptx_runtime.launch(ptx_runtime.load(kernel), kernel, args + (out,))
        expected = F.linear(x.float(), w.float(), b.float() if bias else None).to(dtype)
        torch.testing.assert_close(read(out, outspec), expected, rtol=tol(dtype), atol=tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sq,sk,d,dv,hk,hq,parts,causal,align", [
    (1, 7, 17, 19, 1, 3, 1, False, "upper_left"),
    (1, 9, 33, 17, 2, 4, 4, False, "upper_left"),
    (1, 2, 5, 7, 1, 2, 8, False, "upper_left"),
    (1, 7, 17, 19, 1, 3, 4, True, "upper_left"),
    (3, 7, 17, 19, 1, 2, 1, True, "lower_right"),
    (3, 1, 17, 19, 1, 1, 1, True, "lower_right"),
    (1, 5, 128, 128, 2, 4, 4, False, "upper_left"),
    (1, 2, 256, 256, 1, 1, 1, False, "upper_left"),
])
def test_attention_ptx(ptx_runtime, dtype, sq, sk, d, dv, hk, hq, parts, causal, align):
    torch.manual_seed(41)
    q = torch.randn(1, hq, sq, d, dtype=dtype)*0.3
    k = torch.randn(1, hk, sk, d, dtype=dtype)*0.3
    v = torch.randn(1, hk, sk, dv, dtype=dtype)
    program = attention_program("test_attention", spec(q), spec(k), spec(v), partitions=parts, causal=causal, alignment=align)
    with buffers(ptx_runtime) as (alloc, upload, read):
        args = tuple(upload(t).buffer for t in (q, k, v))
        out = alloc(program.output.nbytes)
        first_out = alloc(program.partial.nbytes) if program.partial else out
        ptx_runtime.launch(ptx_runtime.load(program.first), program.first, args+(first_out,))
        if program.merge:
            ptx_runtime.launch(ptx_runtime.load(program.merge), program.merge, (first_out, out))
        expected = torch.zeros(1, hq, sq, dv)
        scores = q.float() @ k.float().repeat_interleave(hq//hk, 1).transpose(-1, -2) / d**0.5
        for i in range(sq):
            end = sk if not causal else min(sk, max(0, (i+1 if align == "upper_left" else sk-sq+i+1)))
            if end:
                expected[..., i, :] = (scores[..., i:i+1, :end].softmax(-1) @ v.float().repeat_interleave(hq//hk, 1)[..., :end, :]).squeeze(-2)
        torch.testing.assert_close(read(out, program.output), expected.to(dtype), rtol=tol(dtype), atol=tol(dtype))


@pytest.mark.parametrize("width,k", [(1, 1), (17, 4), (129, 8), (1031, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_topk_stable_ptx(ptx_runtime, width, k, dtype):
    torch.manual_seed(49)
    # Repeated values deliberately exercise the documented lowest-index tie rule.
    x = torch.randint(-3, 4, (2, width)).to(dtype)
    with buffers(ptx_runtime) as (alloc, upload, read):
        program = topk("test_topk", spec(x), k)
        dx = upload(x)
        indices, values = alloc(program.indices_nbytes), alloc(program.values_nbytes)
        ptx_runtime.launch(ptx_runtime.load(program.kernel), program.kernel, (dx.buffer, indices, values))
        actual_i = torch.frombuffer(bytearray(ptx_runtime.read(indices, program.indices_nbytes)), dtype=torch.int32).reshape(2, k)
        actual_v = read(values, TensorSpec((2, k)))
        expected_i = torch.argsort(x, dim=-1, descending=True, stable=True)[..., :k]
        assert torch.equal(actual_i.long(), expected_i)
        torch.testing.assert_close(actual_v, x.gather(-1, expected_i).float(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 2])
def test_device_cache_append_decode_reuse(ptx_runtime, dtype, batch):
    torch.manual_seed(51)
    rt = ptx_runtime
    dt = str(dtype).removeprefix("torch.")
    with buffers(rt) as (alloc, upload, read), StaticKVCache(rt, batch=batch, kv_heads=2, capacity=11, head_dim=17, value_dim=19, dtype=dt) as cache:
        decode = cache.prepare_decode(4, partitions=4)
        q = torch.randn(batch, 4, 1, 17, dtype=dtype)*0.3
        dq = upload(q)
        # Empty prefix must return zero without touching uninitialized KV memory.
        empty = decode.run(dq)
        assert torch.count_nonzero(read(empty.buffer, empty.spec)) == 0
        keys, values = [], []
        for tokens in (2, 1, 3):
            k = torch.randn(batch, 2, tokens, 17, dtype=dtype)*0.3
            v = torch.randn(batch, 2, tokens, 19, dtype=dtype)
            keys.append(k); values.append(v)
            cache.append(upload(k), upload(v))
            out = decode.run(dq)
            expected = F.scaled_dot_product_attention(q.float(), torch.cat(keys, 2).float(), torch.cat(values, 2).float(), enable_gqa=True).to(dtype)
            torch.testing.assert_close(read(out.buffer, out.spec), expected, rtol=tol(dtype), atol=tol(dtype))
        assert cache.length == 6 and cache.stats["history_copy_bytes"] == 0
        assert cache.stats["copied_new_kv_bytes"] == batch*(17+19)*2*6*torch.tensor([], dtype=dtype).element_size()
        cache.reset()
        empty = decode.run(dq)
        assert torch.count_nonzero(read(empty.buffer, empty.spec)) == 0


def test_exported_sdpa_runs_direct_ptx(ptx_runtime):
    class M(torch.nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    torch.manual_seed(55)
    args = (torch.randn(1, 4, 1, 17)*0.3, torch.randn(1, 2, 9, 17)*0.3, torch.randn(1, 2, 9, 19))
    plan = compile_exported(torch.export.export(M(), args), decode_partitions=4)
    assert [s.kernel.operation for s in plan.steps] == ["split_decode_partial", "split_decode_merge"]
    with Executor(plan, ptx_runtime) as run, torch.inference_mode():
        torch.testing.assert_close(run(*args), M()(*args), rtol=8e-4, atol=8e-4)


def test_exported_fused_norm_softmax_device_io(ptx_runtime):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.RMSNorm(17, eps=1e-6)
        def forward(self, x, r):
            return self.norm(x + r).softmax(-1)
    torch.manual_seed(56)
    model = M().eval()
    x, r = torch.randn(2, 17)*0.3, torch.randn(2, 17)*0.3
    plan = compile_exported(torch.export.export(model, (x, r)))
    assert [s.kernel.operation for s in plan.steps] == ["residual_rms_norm", "softmax"]
    with buffers(ptx_runtime) as (alloc, upload, read), Executor(plan, ptx_runtime, input_mode="device") as run:
        out = run.run_device(dict(zip(plan.inputs, (upload(x), upload(r)))))
        with torch.inference_mode():
            torch.testing.assert_close(read(out.buffer, out.spec), model(x, r), rtol=8e-4, atol=8e-4)
        assert run.stats["input_upload_bytes"] == run.stats["output_download_bytes"] == 0
