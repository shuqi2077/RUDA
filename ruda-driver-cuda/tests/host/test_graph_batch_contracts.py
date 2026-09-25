"""Source integration guards; these DO NOT execute Rust or a GPU driver."""
from pathlib import Path
import re
import pytest
ROOT = Path(__file__).resolve().parents[3]
def text(name):
    return (ROOT / 'ruda-driver-cuda' / name).read_text()

def body(source, start, end):
    return source.split(start, 1)[1].split(end, 1)[0]

def test_batch_validates_all_layouts_and_storage_before_commit():
    s=text('src/execution/server/graph.rs')
    phase=body(s,'pub(crate) fn graph_update_many','\n}\n')
    assert phase.index('plan_batch(')<phase.index('validate_scalar_update(')<phase.index('// PHASE 2:')
    assert phase.index('// PHASE 2:')<phase.index('command.write_pinned_to_existing')
    assert phase.index('// PHASE 2:')<phase.index('set_scalar_constants')
    assert 'if prepared.is_empty() && !replay { return Ok(()); }' in phase

def test_shadow_signatures_publish_only_after_optional_replay():
    s=text('src/execution/server/graph.rs')
    assert s.index('if replay { entry.native.launch')<s.index('entry.signatures[*index].commit')
    assert 'entry.native.mark_failed(); return Err(err.into())' in s
    assert 'self.graphs.entries.insert(id, entry);' in s

def test_resource_walk_is_outside_changed_node_loop():
    s=body(text('src/execution/server/graph.rs'),'pub(crate) fn graph_update_many','\n}\n')
    assert s.count('for pin in &entry.native.pins')==1
    assert s.index('for pin in &entry.native.pins')<s.index('for (position, descriptor) in prepared')
    assert s.count('entry.native.launch(')==1

def test_single_node_server_method_uses_same_batch_core():
    s=body(text('src/execution/server/graph.rs'),'pub(crate) fn graph_update(','pub(crate) fn graph_update_many')
    assert 'self.graph_update_many(id, vec![(index, dispatch)], replay)' in s

def test_no_mixed_service_calls_inside_public_batch_submission():
    s=body(text('src/graph.rs'),'fn update_many(','fn update(')
    assert s.count('.submit_blocking(')==1
    assert 'server.graph_update_many(id, dispatches, replay)' in s
    assert s.index('same_execution_queue')<s.index('.submit_blocking(')

@pytest.mark.parametrize('method',['update_nodes','update_nodes_and_replay'])
def test_batch_public_methods_are_unsafe(method):
    assert f'pub unsafe fn {method}(' in text('src/graph.rs')

def test_indices_all_validated_before_layout_closure():
    s=text('src/execution/graph_batch.rs')
    assert s.index('check_indices(node_count, &indices)?')<s.index('if validate(')
    assert 'MAX_GRAPH_NODES: usize = 4096' in s
    assert 'if index >= node_count' in s and 'BatchError::Duplicate(index)' in s

def test_tracking_is_opt_in_and_cannot_implicitly_allocate_during_query():
    s=text('src/graph.rs')
    assert 'Self::build_impl(client, nodes, false, None)' in s
    assert 'Self::build_impl(client, nodes, true, None)' in s
    native=text('src/execution/graph.rs')
    q=body(native,'pub fn query_completion(','pub fn wait_completion(')
    assert 'cuEventQuery(event)' in q
    assert 'cuEventCreate' not in q and 'cuStreamQuery' not in q and 'cuEventRecord' not in q

def test_initial_marker_recorded_after_upload():
    s=text('src/execution/server/graph.rs')
    assert s.index('native.finish()?')<s.index('native.enable_completion_tracking()?')
    q=body(text('src/execution/graph.rs'),'pub fn enable_completion_tracking(','pub fn begin_device_work(')
    assert q.index('recorded: false')<q.index('self.record_completion()')

def test_old_marker_invalidated_before_launch_or_metadata_copy():
    s=body(text('src/execution/graph.rs'),'pub fn launch(','pub fn query(')
    assert s.index('self.begin_device_work()')<s.index('cuGraphLaunch(')<s.index('self.record_completion()')
    server=text('src/execution/server/graph.rs')
    assert server.index('entry.native.begin_device_work()')<server.index('command.write_pinned_to_existing')
    assert 'else if uploaded { entry.native.record_completion()?; }' in server

def test_record_failures_poison_before_return():
    s=body(text('src/execution/graph.rs'),'pub fn record_completion(','fn completion_event(')
    assert s.index('completion.recorded = false')<s.index('cuEventRecord')
    assert s.index('self.failed = true')<s.index('return Err(e)')<s.index('completion.recorded = true')

def test_close_fence_is_not_weakened_to_completion_event():
    s=body(text('src/execution/graph.rs'),'pub fn close(','fn destroy_ready(')
    assert 'self.synchronize()?' in s
    assert 'wait_completion' not in s
    s=body(text('src/execution/graph.rs'),'fn destroy_ready(','impl Drop')
    assert s.index('cuGraphExecDestroy')<s.index('cuEventDestroy_v2')<s.index('self.pins.clear()')

def test_legacy_whole_queue_queries_remain_available():
    s=body(text('src/execution/graph.rs'),'pub fn query(','pub fn try_close(')
    assert 'cuStreamQuery(self.stream)' in s
    assert 'cuEventQuery' not in s

def test_no_new_abi_or_default_async_change():
    # This feature is a Rust graph API, not an invented PyTorch graph adapter.
    assert 'RUDA_TORCH_ASYNC' in (ROOT/'ruda-torch/src/lib.rs').read_text()
    assert not list((ROOT/'ruda-driver-cuda/src').rglob('*.cu'))
