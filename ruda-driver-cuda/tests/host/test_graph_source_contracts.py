"""Source wiring/policy checks only. These do NOT compile or execute Rust/PTX."""
from pathlib import Path
import re
import pytest
ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "ruda-driver-cuda"

def text(path): return (ROOT / path).read_text()

def test_graph_uses_native_apis_not_replay_launch_loop():
    source=text("ruda-driver-cuda/src/execution/graph.rs")
    assert all(name in source for name in ("cuGraphCreate", "cuGraphAddKernelNode_v2", "cuGraphInstantiateWithFlags", "cuGraphUpload", "cuGraphLaunch"))
    body=source.split("pub fn launch(",1)[1].split("pub fn synchronize",1)[0]
    assert "launch_kernel" not in body and "cuStreamSynchronize" not in body

def test_graph_objects_drop_before_kernel_modules():
    source=text("ruda-driver-cuda/src/execution/server.rs").split("pub struct CudaServer",1)[1]
    assert source.index("graphs:") < source.index("ctx:")

def test_graph_pins_both_argument_and_device_metadata_buffers():
    source=text("ruda-driver-cuda/src/execution/server.rs")
    assert "bindings.buffers.clone()" in source
    assert "info_binding.as_ref().map(|handle| handle.clone().binding())" in source
    assert "graph.retain(binding, resource.ptr)" in source

def test_graph_waits_before_release_and_keeps_pins_on_failed_shutdown():
    source=text("ruda-driver-cuda/src/execution/graph.rs")
    body=source.split("pub fn close(",1)[1].split("impl Drop",1)[0]
    assert body.index("self.synchronize()?") < body.index("cuGraphExecDestroy") < body.index("self.pins.clear()")
    assert "std::mem::forget(std::mem::take(&mut self.pins))" in source

def test_only_explicit_prepared_path_is_recorded():
    source=text("ruda-driver-cuda/src/execution/server.rs")
    assert "self.launch_checked(kernel, count, bindings, mode, stream_id, None)" in source
    source=text("ruda-driver-cuda/src/execution/server/graph.rs")
    assert "Some(&mut native)" in source
    assert "cuStreamBeginCapture" not in source

def test_graph_preflight_precedes_native_creation():
    source=text("ruda-driver-cuda/src/execution/server/graph.rs")
    assert source.index("validate_dispatch") < source.index("KernelGraph::new")
    assert "ExecutionMode::Checked" in source
    assert "resource.ptr != pin.pointer" in source

def test_prepare_uses_existing_registration_and_no_execute():
    source=text("ruda-kernel-macros/src/ir/generate/launch.rs").split("fn prepare(&self)",1)[1].split("fn launch_body",1)[0]
    assert "self.launch_body()" in source
    assert "launcher.prepare(" in source and "launcher.launch(" not in source

def test_launch_scratch_owns_values_before_forming_addresses():
    source=text("ruda-driver-cuda/src/execution/context/launch.rs")
    assert "resources.iter().map(|resource| resource.ptr)" in source
    assert "memory.binding" not in source
    assert source.index("self.pointers.push(0)") < source.index("self.pointers.iter_mut()")
    assert "self.pointers.clear()" in source and "self.bindings.clear()" in source

def test_cuda_api_version_gates_match_build_cfg():
    build=text("ruda-driver-cuda/build.rs")
    source=text("ruda-driver-cuda/src/execution/graph.rs")
    assert "CUDA_VERSION >= 12000" in build
    assert re.search(r"#\[cfg\(cuda_12000\)\].*?cuGraphAddKernelNode_v2", source, re.S)
    assert re.search(r"#\[cfg\(not\(cuda_12000\)\)\].*?cuGraphAddKernelNode\(", source, re.S)

@pytest.mark.parametrize("case", ["dynamic", "tensor_maps", "same_execution_queue", "4096"])
def test_unsupported_cases_have_explicit_checks(case):
    assert case in text("ruda-driver-cuda/src/graph.rs")
