"""Source wiring checks only. These neither compile nor execute Rust/PTX."""
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[3]
def text(name):return (ROOT/name).read_text()

def test_scalar_prefix_is_captured_before_info_builder_is_consumed():
    s=text('ruda-kernel/src/dsl/compute/launcher.rs').split('pub fn prepare<K:',1)[1].split('/// Launch the kernel.',1)[0]
    assert s.index('scalars.len_aligned()')<s.index('self.into_bindings()')

def test_update_validates_before_mutation():
    s=text('ruda-driver-cuda/src/execution/server/graph.rs').split('pub(crate) fn graph_update',1)[1]
    assert s.index('signature.validate')<s.index('command.write_pinned_to_existing')
    assert s.index('signature.validate')<s.index('set_scalar_constants')
    assert 'compile_kernel' not in s and 'create_with_data' not in s and 'cuGraphInstantiate' not in s

def test_noop_does_not_call_native_update():
    s=text('ruda-driver-cuda/src/execution/server/graph.rs').split('pub(crate) fn graph_update',1)[1]
    # v19 preflights all nodes and commits only selected changed positions.
    assert s.index('plan_batch(')<s.index('for position in changed')
    assert s.index('for position in changed')<s.index('prepared.push((position, descriptor))')
    assert s.index('if prepared.is_empty() && !replay { return Ok(()); }')<s.index('self.ctx.unsafe_set_current')
    assert s.index('for (position, descriptor) in prepared')<s.index('command.write_pinned_to_existing')
    assert s.index('for (position, descriptor) in prepared')<s.index('set_scalar_constants')
    assert 'if replay { entry.native.launch' in s

def test_device_path_writes_scalar_prefix_not_full_metadata():
    s=text('ruda-driver-cuda/src/execution/server/graph.rs')
    assert 'info.data[..dispatch.scalar_words]' in s
    assert 'CopyDescriptor::new(binding, [scalars.len()].into(), [1].into(), 8)' in s
    assert 'binding.offset_end = Some(' in s
    assert 'entry.native.mark_failed(); return Err(err.into())' in s

def test_kernel_param_values_are_owned_not_caller_addresses():
    s=text('ruda-driver-cuda/src/execution/graph.rs')
    assert 'struct StoredNode' in s and 'pointers: Vec<u64>' in s and 'constants: Option<Vec<u64>>' in s
    assert 'cuGraphExecKernelNodeSetParams_v2' in s
    assert 'constants[..scalars.len()].copy_from_slice(scalars)' in s

def test_query_does_not_host_wait():
    s=text('ruda-driver-cuda/src/execution/graph.rs').split('pub fn query(',1)[1].split('pub fn synchronize(',1)[0]
    assert 'cuStreamQuery' in s and 'CUDA_ERROR_NOT_READY => Ok(false)' in s
    assert 'cuStreamSynchronize' not in s and '.synchronize(' not in s

def test_tryclose_busy_keeps_registry_entry():
    s=text('ruda-driver-cuda/src/execution/server/graph.rs')
    assert '&& matches!(&result, Ok(true))' in s
    assert 'if !closed { self.graphs.entries.insert(id, entry); }' in s
    s=text('ruda-driver-cuda/src/graph.rs')
    assert 'if self.command(GraphCommand::TryClose)? { self.id = None;' in s

def test_view_identity_is_not_only_a_device_address():
    s=text('ruda-driver-cuda/src/execution/graph.rs')
    for field in ['binding.memory.id()','binding.stream','binding.size','binding.offset_start','binding.offset_end']:
        assert field in s.split('pub(crate) fn view_key',1)[1].split('#[derive',1)[0]
    assert 'pin_indices.get(&key)' in s
    assert 'self.pins[index].pointer != pointer' in s

@pytest.mark.parametrize('invariant',['kernel != self.kernel','(*x,*y,*z) != self.grid',
    'view_key(a) != view_key(b)','old[old_scalars..] != new[new_scalars..]',
    'old_scalars > old_dynamic','new_scalars > new_dynamic'])
def test_immutable_layout_guards_exist(invariant):
    assert invariant in text('ruda-driver-cuda/src/execution/graph_update.rs')

def test_old_direct_ptx_and_abi_unchanged():
    s=text('ruda-torch/src/lib.rs')
    # v18 does not add bridge ABI calls; source baseline controls the ABI value.
    assert 'RUDA_TORCH_ASYNC' in s
    assert not list((ROOT/'ruda-driver-cuda/src').glob('**/*.cu'))


def test_device_metadata_staging_survives_async_copy():
    s=text('ruda-driver-cuda/src/execution/command.rs')
    method=s.split('pub(crate) fn write_pinned_to_existing(',1)[1].split('/// Allocates a new GPU memory',1)[0]
    assert 'self.reserve_pinned(data.len(), None)' in method
    assert 'self.write_to_gpu(descriptor, staging)' in method
    assert 'current.drop_queue.should_flush()' in method
    assert 'current.drop_queue.flush(|| Fence::new(current.sys))' in method
    assert 'current.drop_queue.push(data)' in s
