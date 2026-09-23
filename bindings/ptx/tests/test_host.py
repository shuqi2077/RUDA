"""Host-only compiler/control-plane tests. These never execute PTX instructions."""
import ctypes
import json
import math
import re
from dataclasses import replace
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from ruda_ptx import Buffer, Executor, TensorSpec, UnsupportedGraph, compile_exported
from ruda_ptx.emitter import elementwise, matmul, rms_norm


class MLP(torch.nn.Module):
    def __init__(self, width=17, hidden=33, dtype=torch.float32):
        super().__init__()
        self.norm = torch.nn.RMSNorm(width, eps=1e-6, dtype=dtype)
        self.gate = torch.nn.Linear(width, hidden, dtype=dtype)
        self.up = torch.nn.Linear(width, hidden, dtype=dtype)
        self.down = torch.nn.Linear(hidden, width, dtype=dtype)

    def forward(self, x):
        y = self.norm(x)
        return self.down(F.silu(self.gate(y)) * self.up(y)) + x


def make_plan(dtype=torch.float32, shape=(3, 17)):
    return compile_exported(torch.export.export(MLP(dtype=dtype).eval(), (torch.zeros(shape, dtype=dtype),)))


@pytest.mark.parametrize("shape", [(), (0,), (-1, 3), (True,), (2**31,), (1.0, 2)])
def test_invalid_shapes(shape):
    with pytest.raises(ValueError):
        TensorSpec(shape)


@pytest.mark.parametrize("dtype", ["int32", "float64", "float8_e4m3fn", "half"])
def test_invalid_dtypes(dtype):
    with pytest.raises(ValueError):
        TensorSpec((16,), dtype)


@pytest.mark.parametrize("dtype,sm,byte_count", [("float32", 70, 4), ("float16", 70, 2), ("bfloat16", 80, 2)])
def test_all_kernel_metadata(dtype, sm, byte_count):
    a = TensorSpec((3, 17), dtype)
    kernels = [elementwise("k_"+op, op, a) for op in ("add", "mul", "silu", "silu_mul")]
    kernels += [rms_norm("norm", a, 1e-6), matmul("mm", a, TensorSpec((17, 35), dtype)),
                matmul("linear", a, TensorSpec((35, 17), dtype), transpose_b=True, bias=True)]
    assert a.nbytes == 3 * 17 * byte_count
    for k in kernels:
        assert k.target_sm == sm
        assert f".target sm_{sm}" in k.ptx
        assert ".version 7.0" in k.ptx and ".address_size 64" in k.ptx
        assert ".visible .entry " + k.name in k.ptx
        assert "nvcc" not in k.ptx and "#include" not in k.ptx
        assert tuple(re.findall(r"\.param \.u64 ([a-z_]+)", k.ptx)) == k.parameters
        labels = set(re.findall(r"^([A-Za-z_][\w]*):", k.ptx, re.M))
        assert set(re.findall(r"\bbra ([A-Za-z_][\w]*);", k.ptx)) <= labels
        assert len(k.digest) == 64 and k.ptx.count("{") == k.ptx.count("}")


@pytest.mark.parametrize("eps", [-1, float("nan"), float("inf"), True, "1e-6", 4e38])
def test_bad_epsilon(eps):
    with pytest.raises(ValueError):
        rms_norm("norm", TensorSpec((3, 17)), eps)


def test_zero_epsilon_allowed():
    assert "sqrt.rn.f32" in rms_norm("norm", TensorSpec((1, 1)), 0).ptx


def test_cache_identity_includes_launch_and_ptx():
    k = elementwise("add", "add", TensorSpec((16,)))
    assert replace(k, block=(128, 1, 1)).digest != k.digest
    assert replace(k, ptx=k.ptx + "\n").digest != k.digest
    assert elementwise("add", "add", TensorSpec((16,))).digest == k.digest


@pytest.mark.parametrize("a,b", [((2, 3), (4, 5)), ((2, 3, 4), (4, 5)), ((1048561, 1), (1, 1))])
def test_matmul_rejection(a, b):
    with pytest.raises(ValueError):
        matmul("mm", TensorSpec(a), TensorSpec(b))


def test_invalid_entry_name():
    with pytest.raises(ValueError):
        elementwise("a;ret;", "add", TensorSpec((2,)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_export_mlp(dtype):
    plan = make_plan(dtype)
    operations = [s.kernel.operation for s in plan.steps]
    assert operations == ["rms_norm", "gated_linear", "linear", "add"]
    assert all(s.dtype == str(dtype).removeprefix("torch.") for s in plan.specs.values())
    assert plan.report()["cpu_compute_fallback"] is False
    assert plan.report()["rust_runtime_connected"] is False
    assert "silu" not in plan.specs  # No activation materialization after fusion.


def test_rank3_linear_flattening():
    plan = make_plan(shape=(2, 3, 17))
    assert plan.specs[plan.outputs[0]].shape == (2, 3, 17)


def test_same_silu_operand_is_not_incorrectly_fused():
    class M(torch.nn.Module):
        def forward(self, x):
            y = F.silu(x)
            return y * y
    p = compile_exported(torch.export.export(M(), (torch.ones(5),)))
    assert [s.kernel.operation for s in p.steps] == ["silu", "mul"]


def test_multiuser_silu_is_kept():
    class M(torch.nn.Module):
        def forward(self, x, y):
            a = F.silu(x)
            return a * y, a + y
    p = compile_exported(torch.export.export(M(), (torch.ones(5), torch.ones(5))))
    assert [s.kernel.operation for s in p.steps] == ["silu", "mul", "add"]


@pytest.mark.parametrize("fn", [lambda x: x.sin(), lambda x: x + 1, lambda x: x[:, 0],
                               lambda x: x.sum(-1), lambda x: x + x[:, :1]])
def test_unsupported_graph_fails_closed(fn):
    class M(torch.nn.Module):
        def forward(self, x):
            return fn(x)
    with pytest.raises(UnsupportedGraph):
        compile_exported(torch.export.export(M(), (torch.ones(3, 17),)))


def test_dynamic_shape_rejected():
    class M(torch.nn.Module):
        def forward(self, x):
            return x + x
    ep = torch.export.export(M(), (torch.ones(3, 17),), dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=9)},))
    with pytest.raises(UnsupportedGraph, match="Dynamic"):
        compile_exported(ep)


def test_mutation_rejected():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.register_buffer("state", torch.ones(3, 17))
        def forward(self, x):
            self.state.add_(x)
            return self.state * x
    ep = torch.export.export(M(), (torch.ones(3, 17),)).run_decompositions({})
    with pytest.raises(UnsupportedGraph):
        compile_exported(ep)


def test_identity_graph_rejected():
    class M(torch.nn.Module):
        def forward(self, x):
            return x
    with pytest.raises(UnsupportedGraph, match="zero PTX"):
        compile_exported(torch.export.export(M(), (torch.ones(3),)))


def test_input_alias_output_rejected():
    class M(torch.nn.Module):
        def forward(self, x):
            return x, x + x
    with pytest.raises(UnsupportedGraph, match="alias"):
        compile_exported(torch.export.export(M(), (torch.ones(3),)))


def test_weights_are_frozen_snapshots():
    model = MLP().eval()
    p = compile_exported(torch.export.export(model, (torch.ones(3, 17),)))
    saved = next(iter(p.constants.values())).clone()
    with torch.no_grad():
        model.norm.weight.fill_(37)
    torch.testing.assert_close(saved, next(iter(p.constants.values())))


def test_workspace_live_ranges_and_capacity():
    p = make_plan()
    assignments, sizes = p.workspace()
    last = {s.output: i for i, s in enumerate(p.steps)}
    for i, s in enumerate(p.steps):
        for n in s.inputs:
            if n in last:
                last[n] = max(last[n], i)
    for n in p.outputs:
        last[n] = len(p.steps)
    for i, a in enumerate(p.steps):
        assert sizes[assignments[a.output]] >= p.specs[a.output].nbytes
        for j, b in enumerate(p.steps):
            if i < j and assignments[a.output] == assignments[b.output]:
                assert last[a.output] < j
    report = p.report()
    assert report["workspace_bytes"] < report["workspace_without_reuse_bytes"]


def test_emit_only_does_not_dump_weights(tmp_path):
    p = make_plan()
    out = tmp_path / "ptx"
    p.write(out)
    files = list(out.iterdir())
    assert len(files) == len(p.steps) + 1
    assert all(f.suffix in {".ptx", ".json"} for f in files)
    assert json.loads((out / "plan.json").read_text())["kernel_format"] == "ptx"
    with pytest.raises(FileExistsError):
        p.write(out)


class RecordingRuntime:
    """Test double, not a GPU/PTX interpreter. Reads return zero bytes only."""
    name = "test_recording_no_compute"
    def __init__(self, fail_launch=False):
        self.calls, self.live, self.next = [], {}, 0
        self.fail_launch = fail_launch
    def allocate(self, nbytes):
        self.next += 1
        b = Buffer(self.next, nbytes, self)
        self.live[b.handle] = b
        self.calls.append(("allocate", nbytes))
        return b
    def free(self, b):
        self.calls.append(("free", b.handle)); del self.live[b.handle]
    def write(self, b, data):
        assert len(data) <= b.nbytes
        self.calls.append(("write", b.handle, len(data)))
    def read(self, b, size):
        assert size <= b.nbytes
        self.calls.append(("read", b.handle, size)); return bytes(size)
    def load(self, k):
        self.calls.append(("load", k.digest)); return k.digest
    def launch(self, handle, kernel, buffers):
        assert handle == kernel.digest and len(buffers) == len(kernel.parameters)
        if self.fail_launch:
            raise RuntimeError("intentional launch failure")
        self.calls.append(("launch", kernel.operation))
    def synchronize(self):
        self.calls.append(("synchronize",))


def test_executor_control_plane_no_reallocation_no_intermediate_readback():
    p, r = make_plan(), RecordingRuntime()
    with Executor(p, r) as e:
        initial_allocs = len([c for c in r.calls if c[0] == "allocate"])
        r.calls.clear()
        with torch.inference_mode():
            e(torch.ones(3, 17)); e(torch.zeros(3, 17))
        assert not any(c[0] == "allocate" for c in r.calls)
        assert len([c for c in r.calls if c[0] == "read"]) == 2
        assert len([c for c in r.calls if c[0] == "launch"]) == 2 * len(p.steps)
        assert len([c for c in r.calls if c[0] == "write"]) == 2
        assert e.stats["calls"] == 2 and initial_allocs > 0
    assert not r.live


def test_executor_rejects_implicit_runtime():
    with pytest.raises(ValueError, match="explicit"):
        Executor(make_plan(), None)


def test_executor_inference_only_and_shape_guards():
    with Executor(make_plan(), RecordingRuntime()) as e:
        with pytest.raises(RuntimeError, match="inference"):
            e(torch.ones(3, 17))
        with torch.inference_mode():
            with pytest.raises(ValueError):
                e(torch.ones(4, 17))
            with pytest.raises(ValueError):
                e(torch.ones(3, 17, dtype=torch.float16))
            with pytest.raises(ValueError):
                e(torch.ones(17, 3).t())
            with pytest.raises(ValueError):
                e()


def test_error_is_not_converted_to_eager_fallback():
    r = RecordingRuntime(fail_launch=True)
    with Executor(make_plan(), r) as e, torch.inference_mode():
        with pytest.raises(RuntimeError, match="intentional"):
            e(torch.ones(3, 17))
        with pytest.raises(RuntimeError, match="closed or failed"):
            e(torch.ones(3, 17))


def test_duplicate_computed_outputs_preserve_alias():
    class M(torch.nn.Module):
        def forward(self, x):
            y = x + x
            return {"a": y, "b": y}
    p = compile_exported(torch.export.export(M(), (torch.ones(3),)))
    r = RecordingRuntime()
    with Executor(p, r) as e, torch.inference_mode():
        out = e(torch.ones(3))
        assert out["a"] is out["b"]
        assert len([c for c in r.calls if c[0] == "read"]) == 1


def test_byte_transport_preserves_bfloat16_bits():
    x = torch.tensor([1.0, -2.0, float("nan")], dtype=torch.bfloat16)
    data = Executor._tensor_bytes(x)
    restored = torch.frombuffer(bytearray(data), dtype=torch.bfloat16)
    assert torch.equal(x.view(torch.int16), restored.view(torch.int16))


def test_missing_driver_is_an_explicit_error(monkeypatch):
    from ruda_ptx.nvidia_driver import NvidiaDriverRuntime, DriverError
    def unavailable(*args, **kwargs):
        raise OSError("missing driver")
    monkeypatch.setattr(ctypes, "CDLL", unavailable)
    with pytest.raises(DriverError, match="no CPU fallback"):
        NvidiaDriverRuntime()


def test_core_has_no_cuda_extension_or_inductor_execution():
    import ast
    import ruda_ptx
    root = Path(ruda_ptx.__file__).parent
    forbidden = {"CUDAExtension", "load_inline", "compile_ptx", "lookup_backend"}
    for file in root.glob("*.py"):
        tree = ast.parse(file.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                assert n.attr not in forbidden
                assert not (isinstance(n.value, ast.Name) and n.value.id == "torch" and n.attr == "cuda")
