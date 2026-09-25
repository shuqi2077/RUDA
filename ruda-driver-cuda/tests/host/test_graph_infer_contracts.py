"""Source wiring guards only, not a Rust compiler or GPU test."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3] / 'ruda-driver-cuda'
def text(p): return (ROOT/p).read_text()

def test_new_inference_is_an_explicit_public_api():
    s=text('src/graph.rs')
    assert 'pub unsafe fn build_inferred(' in s and 'pub unsafe fn build_inferred_tracked(' in s
    assert 'server.graph_build_inferred(stream, dispatches, track_completion)' in s
    assert 'Same requirements as `build_inferred`' in s

def test_inference_uses_real_ir_before_native_construction():
    s=text('src/execution/server/graph.rs')
    assert 'dispatch.task.kernel_definition()' in s
    block=s.split('pub(crate) fn graph_build_inferred',1)[1].split('pub(crate) fn graph_command',1)[0]
    assert block.index('collect_graph_accesses') < block.index('GraphTopology::infer') < block.index('self.graph_build')

def test_legacy_defaults_still_chain_and_validate_explicit_dag():
    s=text('src/graph.rs')
    assert 'None => GraphTopology::chain(nodes.len())' in s
    assert 'Some(parents) => GraphTopology::new(nodes.len(), parents)' in s
    assert 'Self::build_impl(client, nodes, false, None)' in s

def test_interval_sweep_filters_disjoint_and_readonly_pairs():
    s=text('src/execution/graph_topology.rs')
    assert 'expire(&mut readers, b.start)' in s and 'expire(&mut writers, b.start)' in s
    assert 'last.writable == a.writable' in s and 'a.start <= last.end' in s
    assert 'if b.writable' in s and 'MAX_INFERENCE_CHECKS' in s

def test_inference_prunes_only_already_reachable_parents():
    s=text('src/execution/graph_topology.rs')
    assert 'for word in (0..words).rev()' in s and 'candidates.leading_zeros()' in s
    assert 'hazards[node][word] & !covered[word]' in s and 'zip(&ancestors[parent])' in s

def test_inference_does_not_run_on_replay_or_scalar_update():
    s=text('src/execution/server/graph.rs').split('pub(crate) fn graph_command',1)[1]
    assert 'GraphTopology::infer' not in s and 'collect_graph_accesses' not in s

def test_direct_ptx_test_and_example_registered():
    s=text('Cargo.toml')
    assert 'name = "graph-infer"' in s and 'path = "tests/graph_infer.rs"' in s
    assert 'name = "native-graph-infer"' in s

def test_gpu_acceptance_has_no_cpu_substitute():
    s=text('tests/graph_infer.rs')
    assert 'CpuRuntime' not in s and 'RUDA_GRAPH_INFER_RUNTIME,CudaRuntime,direct-ptx' in s
    assert s.count('#[ignore') == 1 and 'graph_infer_benchmark' in s
