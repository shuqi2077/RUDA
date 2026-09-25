"""Source wiring checks only: these do not compile or execute production Rust."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[3]
def source(path):return (ROOT/'ruda-driver-cuda'/path).read_text()

def test_public_api_keeps_legacy_chain_explicit():
    text=source('src/graph.rs')
    assert 'Self::build_impl(client, nodes, false, None)' in text
    assert 'Self::build_impl(client, nodes, true, None)' in text
    assert 'None => GraphTopology::chain(nodes.len())' in text

def test_dag_api_uses_explicit_parent_list_and_optional_tracking():
    text=source('src/graph.rs')
    assert 'pub unsafe fn build_dag(' in text and 'pub unsafe fn build_dag_tracked(' in text
    assert 'Some(dependencies)' in text and 'dependencies: Vec<Vec<usize>>' in text

def test_topology_validation_precedes_device_submission():
    text=source('src/graph.rs').split('unsafe fn build_impl',1)[1]
    assert text.index('GraphTopology::new')<text.index('.submit_blocking')

def test_real_ir_not_unchecked_caller_annotation():
    text=source('src/execution/server/graph.rs')
    assert 'dispatch.task.kernel_definition()' in text
    assert 'argument.visibility != Visibility::Read' in text
    assert 'definition.buffers.len() != dispatch.arguments.buffers.len()' in text
    assert 'definition.options.cluster_dim.is_some()' in text

def test_hazards_rejected_before_any_native_construction():
    text=source('src/execution/server/graph.rs').split('pub(crate) fn graph_build',1)[1]
    assert text.index('validate_dag_accesses')<text.index('KernelGraph::new')<text.index('self.launch_checked')

def test_identity_and_view_bounds_are_checked():
    text=source('src/execution/server/graph.rs')
    assert 'allocations.entry(binding.memory.id())' in text
    assert 'size != binding.size' in text and 'start > end' in text
    assert '.checked_sub(binding.offset_end.unwrap_or(0))' in text

def test_actual_parent_handles_reach_driver():
    text=source('src/execution/graph.rs')
    assert 'last_node' not in text
    assert 'self.dependencies.get(self.nodes)' in text
    assert 'self.retained_nodes.get(index).map(|node| node.node)' in text
    assert 'parent_handles.as_ptr()' in text and 'let count = parent_handles.len()' in text
    assert 'cuGraphAddKernelNode_v2(&mut node, self.graph, dependencies, count, &params)' in text

def test_incomplete_native_graph_cannot_instantiate():
    text=source('src/execution/graph.rs')
    assert 'self.nodes != self.dependencies.len()' in text

def test_no_hazard_scan_in_replay_hot_path():
    text=source('src/execution/server/graph.rs').split('pub(crate) fn graph_command',1)[1]
    assert 'validate_dag_accesses' not in text

def test_scalar_update_keeps_original_topology():
    text=source('src/execution/server/graph.rs').split('pub(crate) fn graph_update_many',1)[1]
    assert 'entry.signatures[index]' in text
    assert 'dependencies =' not in text and 'cuGraphAdd' not in text

def test_planner_is_dependency_free_and_bounded():
    text=source('src/execution/graph_topology.rs')
    assert 'use cudarc' not in text and 'use ruda' not in text
    assert 'MAX_ALIAS_CHECKS' in text and 'MAX_EDGES' in text and 'MAX_ACCESSES' in text
    assert 'a.start < b.end && b.start < a.end' in text
    assert 'self.ordered(first, second)' in text

def test_diagnostics_never_emit_device_addresses():
    text=source('src/execution/graph_topology.rs').split('pub fn to_dot',1)[1].split('#[cfg(test)]',1)[0]
    assert '0x' not in text and 'pointer' not in text and 'scalar' not in text
    assert 'n{parent} -> n{node}' in text

def test_all_new_device_tests_are_registered():
    manifest=source('Cargo.toml')
    assert 'name = "graph-dag"' in manifest and 'path = "tests/graph_dag.rs"' in manifest
    assert 'name = "native-graph-dag"' in manifest

def test_fixture_tests_are_not_ignored_device_acceptance():
    text=source('tests/graph_dag.rs')
    assert 'CpuRuntime' not in text and 'if let Ok(' not in text
    assert text.count('#[ignore')==1
    assert 'graph_dag_benchmark' in text
