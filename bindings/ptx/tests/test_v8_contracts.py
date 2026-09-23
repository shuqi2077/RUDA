"""Guards, lifetimes and generated-code invariants, not hardware validation."""
import re
import pytest
import torch
import torch.nn.functional as F
from ruda_ptx import TensorSpec, DeviceTensor, Executor, StaticKVCache, compile_exported, UnsupportedGraph
from ruda_ptx.attention import attention_program
from ruda_ptx.reductions import rms_norm, softmax
from ruda_ptx.emitter import rms_norm_reference
from ruda_ptx.cache_kernels import kv_append
from ruda_ptx.rotary import rope
from ruda_ptx.selection import topk
from test_host import RecordingRuntime


class ValidatingRecorder(RecordingRuntime):
    def validate_buffer(self, b):
        if b.owner is not self or self.live.get(b.handle) is not b:
            raise ValueError("foreign/freed buffer")
    def launch_many(self, calls):
        self.calls.append(("batch",))
        for h, k, b in calls:
            for buffer in b:
                self.validate_buffer(buffer)
            self.launch(h, k, b)


def test_rms_barrier_and_shared_reduction():
    wide = TensorSpec((2, 4096))
    old, new = rms_norm_reference("old", wide, 1e-6), rms_norm("new", wide, 1e-6)
    assert old.ptx.count("bar.sync 0;") == 9
    assert new.ptx.count("bar.sync 0;") == 2
    assert "partials[32]" in new.ptx and "sums[1024]" in old.ptx
    assert rms_norm("narrow", TensorSpec((4, 128)), 1e-6).ptx.count("bar.sync") == 0


def test_softmax_has_guard_before_scratch_reuse():
    kernel = softmax("sm", TensorSpec((1, 2048)))
    assert kernel.ptx.count("bar.sync 0;") == 5


def test_attention_workspace_independent_of_context_length():
    q = TensorSpec((1, 32, 1, 128), "float16")
    programs = [attention_program("attn", q, TensorSpec((1, 8, length, 128), "float16"),
                TensorSpec((1, 8, length, 128), "float16"), partitions=8) for length in (512, 32768)]
    assert programs[0].workspace_bytes == programs[1].workspace_bytes == 32*8*(128+2)*4
    assert all(".shared" not in k.ptx for p in programs for k in p.kernels)
    one = attention_program("one", q, TensorSpec((1, 8, 9, 128), "float16"), TensorSpec((1, 8, 9, 128), "float16"))
    assert one.workspace_bytes == 0


@pytest.mark.parametrize("options", [{"partitions": 0}, {"partitions": 33}, {"partitions": True},
    {"scale": float('nan')}, {"scale": float('inf')}, {"scale": True}, {"causal": 1},
    {"alignment": "automatic"}, {"dynamic_length": 1}])
def test_attention_flags_fail_closed(options):
    with pytest.raises((TypeError, ValueError)):
        attention_program("a", TensorSpec((1, 4, 1, 17)), TensorSpec((1, 2, 7, 17)), TensorSpec((1, 2, 7, 19)), **options)


@pytest.mark.parametrize("dims", [((1, 3, 1, 17),(1, 2, 7, 17),(1, 2, 7, 19)),
    ((1,4,1,257),(1,2,7,257),(1,2,7,19)), ((1,4,1,17),(1,2,7,17),(2,2,7,19)),
    ((1,4,1,17),(1,2,7,17),(1,2,8,19))])
def test_attention_bad_shapes(dims):
    with pytest.raises(ValueError):
        attention_program("a", *(TensorSpec(d) for d in dims))


@pytest.mark.parametrize("rd", [0, -2, 3, 20, True])
def test_bad_rotary_dim(rd):
    with pytest.raises(ValueError):
        rope("r", TensorSpec((1, 2, 3, 17)), rd)


@pytest.mark.parametrize("k", [0, -1, 9, True])
def test_bad_topk(k):
    with pytest.raises(ValueError):
        topk("k", TensorSpec((1, 17)), k)


def test_cache_copy_and_length_update_are_ordered_separate_kernels():
    copy, advance = kv_append("append", TensorSpec((2, 3, 1, 17)), TensorSpec((2, 3, 1, 19)), 1024)
    assert copy.operation == "kv_append_copy" and advance.operation == "kv_append_advance"
    assert "st.global.u32 [%rd4]" not in copy.ptx
    assert "st.global.u32 [%rd0]" in advance.ptx


def test_cache_hot_path_no_allocation_no_host_transfer_no_recompile():
    r = ValidatingRecorder()
    with StaticKVCache(r, batch=1, kv_heads=2, capacity=7, head_dim=17) as cache:
        decode = cache.prepare_decode(4, partitions=4)
        kn = TensorSpec((1, 2, 1, 17), "float16")
        qn = TensorSpec((1, 4, 1, 17), "float16")
        k, v, q = (DeviceTensor(r.allocate(s.nbytes), s) for s in (kn, kn, qn))
        r.calls.clear()
        for _ in range(3):
            cache.append(k, v)
            decode.run(q)
        assert not any(c[0] in {"allocate", "write", "read", "load", "synchronize"} for c in r.calls)
        assert len([c for c in r.calls if c[0] == "launch"]) == 12
        assert cache.stats["history_copy_bytes"] == 0
        assert cache.length == 3
        for t in (k, v, q):
            r.free(t.buffer)


def test_cache_overflow_rejected_before_any_launch_and_can_continue():
    r = ValidatingRecorder()
    with StaticKVCache(r, batch=1, kv_heads=1, capacity=1, head_dim=2) as c:
        s = TensorSpec((1, 1, 1, 2), "float16")
        k, v = DeviceTensor(r.allocate(s.nbytes), s), DeviceTensor(r.allocate(s.nbytes), s)
        c.append(k, v)
        r.calls.clear()
        with pytest.raises(ValueError, match="capacity"):
            c.append(k, v)
        assert not r.calls and c.length == 1
        c.reset()
        c.append(k, v)
        r.free(k.buffer); r.free(v.buffer)


def test_device_cache_rejects_foreign_buffers():
    a, b = ValidatingRecorder(), ValidatingRecorder()
    with StaticKVCache(a, batch=1, kv_heads=1, capacity=2, head_dim=2) as c:
        s = TensorSpec((1, 1, 1, 2), "float16")
        k = DeviceTensor(b.allocate(s.nbytes), s)
        v = DeviceTensor(a.allocate(s.nbytes), s)
        with pytest.raises(ValueError, match="foreign"):
            c.append(k, v)
        assert c.length == 0
        a.free(v.buffer); b.free(k.buffer)


def test_cache_poisoned_on_partial_launch_failure():
    r = ValidatingRecorder()
    with StaticKVCache(r, batch=1, kv_heads=1, capacity=2, head_dim=2) as c:
        s = TensorSpec((1, 1, 1, 2), "float16")
        k, v = DeviceTensor(r.allocate(s.nbytes), s), DeviceTensor(r.allocate(s.nbytes), s)
        r.fail_launch = True
        with pytest.raises(RuntimeError, match="intentional"):
            c.append(k, v)
        r.fail_launch = False
        with pytest.raises(RuntimeError, match="failed"):
            c.append(k, v)
        r.free(k.buffer); r.free(v.buffer)


class Add(torch.nn.Module):
    def forward(self, x):
        return x + x


def test_device_executor_chain_no_internal_host_transfers():
    r = ValidatingRecorder()
    p = compile_exported(torch.export.export(Add(), (torch.ones(2, 17),)))
    s = TensorSpec((2, 17))
    x = DeviceTensor(r.allocate(s.nbytes), s)
    with Executor(p, r, input_mode="device") as a, Executor(p, r, input_mode="device") as b:
        r.calls.clear()
        y = a.run_device({p.inputs[0]: x})
        z = b.run_device({p.inputs[0]: y})
        assert z.spec == s
        assert not any(c[0] in {"read", "write", "allocate", "load", "synchronize"} for c in r.calls)
        with pytest.raises(ValueError, match="aliases"):
            a.run_device({p.inputs[0]: y})
        assert a.stats["input_upload_bytes"] == a.stats["output_download_bytes"] == 0
    r.free(x.buffer)
    assert not r.live


def test_attention_export_causal_remains_upper_left():
    class M(torch.nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    args = (torch.randn(1,2,1,17), torch.randn(1,2,7,17), torch.randn(1,2,7,19))
    p = compile_exported(torch.export.export(M(), args))
    assert "add.u32 %r13, %r4, 1;" in p.steps[0].kernel.ptx
    assert "sub.s32 %r13, %r14" not in p.steps[0].kernel.ptx


def test_prefill_requires_opt_in():
    class M(torch.nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v)
    args = (torch.randn(1,2,3,17), torch.randn(1,2,7,17), torch.randn(1,2,7,19))
    ep = torch.export.export(M(), args)
    with pytest.raises(UnsupportedGraph, match="prefill"):
        compile_exported(ep)
    assert compile_exported(ep, allow_streaming_prefill=True).steps[0].kernel.operation == "online_attention"


def test_residual_with_another_user_is_not_eliminated():
    class M(torch.nn.Module):
        def forward(self, x, r, w):
            z = x + r
            return F.rms_norm(z, (17,), w, 1e-6), z
    args = (torch.randn(1,17), torch.randn(1,17), torch.ones(17))
    p = compile_exported(torch.export.export(M(), args))
    assert [s.kernel.operation for s in p.steps] == ["add", "rms_norm"]


def test_decode_linear_policy_can_be_disabled():
    m = torch.nn.Linear(17, 19).eval()
    ep = torch.export.export(m, (torch.randn(1,17),))
    a, b = compile_exported(ep), compile_exported(ep, use_decode_linear=False)
    assert a.steps[0].kernel.block == (128,1,1)
    assert b.steps[0].kernel.block == (16,16,1)
