"""Compiler/session contract tests; no GPU performance claims."""
import pytest
import torch
import torch.nn.functional as F
from ruda_ptx import TensorSpec, TopKSession, DeviceTensor, compile_exported
from ruda_ptx.decode_linear import linear_decode, gated_linear
from ruda_ptx.partitioned_selection import partitioned_topk
from ptx_debug import DebugRuntime


class Gates(torch.nn.Module):
    def __init__(self, *, escape=False, separate=False, hidden=17):
        super().__init__()
        self.gate = torch.nn.Linear(7, hidden)
        self.up = torch.nn.Linear(7, hidden)
        self.escape, self.separate = escape, separate
    def forward(self, x):
        g = self.gate(x)
        y = F.silu(g)*self.up(x+x if self.separate else x)
        return (y, g) if self.escape else y


@pytest.mark.parametrize("escape,separate,m", [(True, False, 1), (False, True, 1), (False, False, 5)])
def test_no_incorrect_gated_fusion(escape, separate, m):
    ep = torch.export.export(Gates(escape=escape, separate=separate), (torch.ones(m, 7),))
    p = compile_exported(ep)
    assert not any(s.kernel.operation == "gated_linear" for s in p.steps)


@pytest.mark.parametrize("option", ["fuse_gated_decode", "use_decode_linear"])
def test_fusion_opt_out(option):
    ep = torch.export.export(Gates(), (torch.ones(1, 7),))
    p = compile_exported(ep, **{option: False})
    assert not any(s.kernel.operation == "gated_linear" for s in p.steps)


def test_decode_tile_policy_and_override():
    ep = torch.export.export(torch.nn.Linear(7, 1024), (torch.ones(1, 7),))
    default = compile_exported(ep).steps[0].kernel
    one = compile_exported(ep, decode_outputs_per_warp=1).steps[0].kernel
    four = compile_exported(ep, decode_outputs_per_warp=4).steps[0].kernel
    assert default.grid[0] == one.grid[0]//2 == four.grid[0]*2


@pytest.mark.parametrize("tile", [0, 3, True, 2.0])
def test_bad_tile(tile):
    with pytest.raises(ValueError):
        linear_decode("x", TensorSpec((1, 7)), TensorSpec((17, 7)), outputs_per_warp=tile)


@pytest.mark.parametrize("k,parts", [(0, 4), (9, 4), (True, 4), (1, 0), (1, 65), (1, False)])
def test_topk_invalid(k, parts):
    with pytest.raises(ValueError):
        partitioned_topk("x", TensorSpec((1, 33)), k, partitions=parts)


def test_gated_rejects_prefill():
    with pytest.raises(ValueError):
        gated_linear("x", TensorSpec((5, 7)), TensorSpec((17, 7)))


def test_warm_selection_no_allocation_transfer_or_jit():
    runtime = DebugRuntime()
    source = runtime.allocate(17*4)
    runtime.write(source, torch.arange(17, dtype=torch.float32).numpy().tobytes())
    x = DeviceTensor(source, TensorSpec((1, 17)))
    with TopKSession(runtime, x.spec, 4, partitions=3) as session:
        runtime.calls.clear()
        session.run(x)
        session.run(x)
        assert [c[0] for c in runtime.calls] == ["launch"]*4
        assert session.program.workspace_nbytes == 1*3*4*8
    runtime.free(source)
    assert not runtime.buffers


def test_failed_session_poison_and_close(monkeypatch):
    runtime = DebugRuntime()
    source = runtime.allocate(4)
    runtime.write(source, b'\0'*4)
    with TopKSession(runtime, TensorSpec((1,)), 1, partitions=1) as session:
        def fail(*_):
            raise RuntimeError("injected submit failure")
        monkeypatch.setattr(runtime, "launch_many", fail)
        with pytest.raises(RuntimeError, match="injected"):
            session.run(DeviceTensor(source, TensorSpec((1,))))
        with pytest.raises(RuntimeError, match="failed"):
            session.run(DeviceTensor(source, TensorSpec((1,))))
    runtime.free(source)
    assert not runtime.buffers
