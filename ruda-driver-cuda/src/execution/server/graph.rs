use super::*;
use crate::execution::graph::{KernelGraph, error};
use crate::graph::{GraphCommand, GraphDispatch};
use crate::execution::graph_update::NodeSignature;
use crate::execution::graph_batch::plan_batch;
use crate::execution::graph_topology::{BufferAccess, GraphTopology, MAX_ACCESSES};
use ruda_core::kernel::Visibility;

/// Inspect real kernel IR, not caller-supplied read/write promises. Unknown IR
/// is refused for DAG builds; legacy ordered builds keep their prior behavior.
fn collect_graph_accesses(dispatches: &[GraphDispatch])
    -> Result<Vec<Vec<BufferAccess>>, ServerError>
{
    let mut allocations = HashMap::new();
    let mut accesses = Vec::with_capacity(dispatches.len());
    let mut total = 0usize;
    for (index, dispatch) in dispatches.iter().enumerate() {
        total = total.checked_add(dispatch.arguments.buffers.len())
            .ok_or_else(|| error("DAG access count overflow"))?;
        if total > MAX_ACCESSES { return Err(error("DAG has too many buffer accesses")); }
        let definition = dispatch.task.kernel_definition()
            .ok_or_else(|| error(format!("DAG node {index}: kernel IR is required for access validation")))?;
        if definition.buffers.len() != dispatch.arguments.buffers.len()
            || !definition.tensor_maps.is_empty() || definition.options.cluster_dim.is_some() {
            return Err(error(format!("DAG node {index}: unsupported IR binding/cluster layout")));
        }
        let mut node = Vec::with_capacity(definition.buffers.len());
        for (argument, binding) in definition.buffers.iter().zip(&dispatch.arguments.buffers) {
            let next_id = allocations.len();
            let (allocation, size) = *allocations.entry(binding.memory.id())
                .or_insert((next_id, binding.size));
            if size != binding.size { return Err(error("DAG allocation has inconsistent base sizes")); }
            let start = binding.offset_start.unwrap_or(0);
            let end = binding.size.checked_sub(binding.offset_end.unwrap_or(0))
                .ok_or_else(|| error("DAG view exceeds allocation"))?;
            if start > end { return Err(error("DAG view has invalid offsets")); }
            node.push(BufferAccess { allocation, start, end,
                writable: argument.visibility != Visibility::Read });
        }
        accesses.push(node);
    }
    Ok(accesses)
}

fn validate_dag_accesses(topology: &GraphTopology, dispatches: &[GraphDispatch])
    -> Result<(), ServerError>
{
    let accesses = collect_graph_accesses(dispatches)?;
    topology.validate_accesses(&accesses).map_err(|e| error(format!("DAG memory dependency: {e:?}")))
}

#[derive(Debug)]
pub(super) struct GraphRegistry {
    next_id: u64,
    entries: HashMap<u64, GraphEntry>,
}
impl Default for GraphRegistry {
    fn default() -> Self { Self { next_id: 1, entries: HashMap::new() } }
}
#[derive(Debug)]
struct GraphEntry { stream_id: StreamId, native: KernelGraph, signatures: Vec<NodeSignature> }

impl CudaServer {
    pub(crate) fn graph_build(
        &mut self, stream_id: StreamId, dispatches: Vec<GraphDispatch>, track_completion: bool,
        topology: GraphTopology, check_accesses: bool,
    ) -> Result<(u64, usize), ServerError> {
        // Validate every node before compiling/loading anything or preparing GPU metadata.
        if dispatches.is_empty() || dispatches.len() > 4096 {
            return Err(error("native graph requires 1..=4096 explicit kernel nodes"));
        }
        let signatures = dispatches.iter().map(NodeSignature::new).collect::<Result<Vec<_>, _>>()?;
        for dispatch in &dispatches {
            crate::graph::validate_dispatch(&dispatch.count, &dispatch.arguments)?;
        }
        if topology.parents().len() != dispatches.len() { return Err(error("DAG node count mismatch")); }
        // Before native graph creation, kernel compilation or metadata upload.
        if check_accesses { validate_dag_accesses(&topology, &dispatches)?; }
        if !self.ctx.timestamps.is_empty() {
            return Err(error("finish active runtime profiling before graph construction"));
        }
        let api = crate::query_driver_api_version().map_err(|e| error(e.to_string()))?;
        let minimum = if cfg!(cuda_12000) { 12000 } else { 11040 };
        if api < minimum { return Err(error("driver version is too old for compiled native graph bindings")); }
        let next = self.graphs.next_id.checked_add(1).ok_or_else(|| error("graph identifier overflow"))?;
        let physical = {
            let mut command = self.command_no_inputs(stream_id, StreamErrorMode { ignore: false, flush: true })?;
            command.streams.current().sys
        };
        let mut native = KernelGraph::new(self.ctx.context, physical, topology.into_parents())?;
        for dispatch in dispatches {
            // The same compiler, metadata layout, GPU allocator, kernel cache and
            // function handles are used by ordinary launches and graph nodes.
            self.launch_checked(dispatch.task, dispatch.count, dispatch.arguments,
                ExecutionMode::Checked, stream_id, Some(&mut native))?;
        }
        native.finish()?;
        if track_completion { native.enable_completion_tracking()?; }
        let id = self.graphs.next_id;
        let nodes = native.nodes;
        self.graphs.next_id = next;
        self.graphs.entries.insert(id, GraphEntry { stream_id, native, signatures });
        Ok((id, nodes))
    }

    /// Infer dependencies before compiling, uploading metadata or creating any
    /// driver graph. Access declarations come from the exact prepared kernel IR.
    pub(crate) fn graph_build_inferred(
        &mut self, stream_id: StreamId, dispatches: Vec<GraphDispatch>, track_completion: bool,
    ) -> Result<(u64, usize, Vec<Vec<usize>>), ServerError> {
        if dispatches.is_empty() || dispatches.len() > 4096 {
            return Err(error("native graph requires 1..=4096 explicit kernel nodes"));
        }
        for dispatch in &dispatches {
            crate::graph::validate_dispatch(&dispatch.count, &dispatch.arguments)?;
        }
        let accesses = collect_graph_accesses(&dispatches)?;
        let topology = GraphTopology::infer(&accesses)
            .map_err(|e| error(format!("graph dependency inference: {e:?}")))?;
        let dependencies = topology.parents().to_vec();
        // Inference already proves all declared overlap hazards. Avoid running
        // a second quadratic/interval scan under the explicit-DAG work budget.
        let (id, nodes) = self.graph_build(stream_id, dispatches, track_completion, topology, false)?;
        Ok((id, nodes, dependencies))
    }

    pub(crate) fn graph_command(&mut self, id: u64, operation: GraphCommand) -> Result<bool, ServerError> {
        // Remove while resolving stream dependencies, avoiding overlapping borrows
        // of the server. On every error the entry is reinserted for explicit close.
        let mut entry = self.graphs.entries.remove(&id).ok_or_else(|| error("unknown/closed RUDA graph"))?;
        let result = (|| {
            self.ctx.unsafe_set_current().map_err(|e| error(format!("graph context: {e:?}")))?;
            match operation {
                GraphCommand::Replay => {
                    let pins = &entry.native.pins;
                    let mut command = self.command(entry.stream_id, pins.iter().map(|p| &p.binding),
                        StreamErrorMode { ignore: false, flush: true })?;
                    // Refuse replay if an allocator ever relocates a retained view.
                    // Do not silently patch pointer values or race a prior launch.
                    for pin in pins {
                        let resource = command.resource(pin.binding.clone())?;
                        if resource.ptr != pin.pointer {
                            return Err(error("native graph buffer address changed; rebuild the graph"));
                        }
                    }
                    entry.native.launch(command.streams.current().sys).map(|_| true)
                }
                GraphCommand::Synchronize => {
                    // Propagate deferred RUDA launch errors, not just driver errors.
                    let _ = self.command_no_inputs(entry.stream_id,
                        StreamErrorMode { ignore: false, flush: true })?;
                    entry.native.synchronize().map(|_| true)
                },
                GraphCommand::Query | GraphCommand::TryClose => {
                    let _ = self.command_no_inputs(entry.stream_id,
                        StreamErrorMode { ignore: false, flush: true })?;
                    if matches!(operation, GraphCommand::TryClose) { entry.native.try_close() }
                    else { entry.native.query() }
                },
                GraphCommand::QueryCompletion | GraphCommand::WaitCompletion => {
                    let _ = self.command_no_inputs(entry.stream_id,
                        StreamErrorMode { ignore: false, flush: true })?;
                    if matches!(operation, GraphCommand::WaitCompletion) {
                        entry.native.wait_completion().map(|_| true)
                    } else { entry.native.query_completion() }
                },
                GraphCommand::Close => entry.native.close().map(|_| true),
            }
        })();
        let closed = matches!(operation, GraphCommand::Close | GraphCommand::TryClose)
            && matches!(&result, Ok(true));
        if !closed { self.graphs.entries.insert(id, entry); }
        result
    }

    pub(crate) fn graph_update(&mut self, id: u64, index: usize, dispatch: GraphDispatch, replay: bool)
        -> Result<(), ServerError>
    {
        self.graph_update_many(id, vec![(index, dispatch)], replay)
    }

    pub(crate) fn graph_update_many(&mut self, id: u64,
        updates: Vec<(usize, GraphDispatch)>, replay: bool) -> Result<(), ServerError>
    {
        let mut entry = self.graphs.entries.remove(&id).ok_or_else(|| error("unknown/closed RUDA graph"))?;
        let result = (|| {
            entry.native.check_executable()?;
            // PHASE 1: all node layouts and all scalar destinations are checked
            // before ANY native parameter update, pinned copy or graph launch.
            let changed = plan_batch(entry.signatures.len(), &updates, |index, dispatch| {
                let signature = &entry.signatures[index];
                signature.validate(dispatch)
            }).map_err(|e| error(format!("graph batch preflight: {e:?}")))?;
            let mut prepared = Vec::with_capacity(changed.len());
            for position in changed {
                let (index, dispatch) = &updates[position];
                let scalars = &dispatch.arguments.info.data[..dispatch.scalar_words];
                let descriptor = if let Some(mut binding) = entry.native.validate_scalar_update(*index, scalars.len())? {
                    let bytes = scalars.len().checked_mul(8).ok_or_else(|| error("scalar byte count overflow"))?;
                    let available = binding.size.checked_sub(binding.offset_start.unwrap_or(0))
                        .and_then(|n| n.checked_sub(binding.offset_end.unwrap_or(0)))
                        .ok_or_else(|| error("invalid fixed metadata allocation view"))?;
                    let bytes = u64::try_from(bytes).map_err(|_| error("scalar byte count overflow"))?;
                    if bytes > available { return Err(error("scalar update exceeds fixed metadata allocation")); }
                    binding.offset_end = Some(binding.size - binding.offset_start.unwrap_or(0) - bytes);
                    Some(CopyDescriptor::new(binding, [scalars.len()].into(), [1].into(), 8))
                } else { None };
                prepared.push((position, descriptor));
            }
            // A pure no-op still validated the complete batch, but requires no
            // device context switch, resource walk, upload or native call.
            if prepared.is_empty() && !replay { return Ok(()); }
            self.ctx.unsafe_set_current().map_err(|e| error(format!("graph update context: {e:?}")))?;
            let mut command = self.command(entry.stream_id, entry.native.pins.iter().map(|p| &p.binding),
                StreamErrorMode { ignore: false, flush: true })?;
            // Once per BATCH, not once per changed node.
            for pin in &entry.native.pins {
                if command.resource(pin.binding.clone())?.ptr != pin.pointer {
                    return Err(error("native graph buffer address changed; rebuild the graph"));
                }
            }
            // PHASE 2: no remaining fallible layout checks. A driver/copy error
            // poisons the graph; a partially committed batch MUST NOT replay.
            let mut uploaded = false;
            for (position, descriptor) in prepared {
                let (index, dispatch) = &updates[position];
                let scalars = &dispatch.arguments.info.data[..dispatch.scalar_words];
                if let Some(descriptor) = descriptor {
                    entry.native.begin_device_work();
                    if let Err(err) = command.write_pinned_to_existing(descriptor, bytemuck::cast_slice(scalars)) {
                        entry.native.mark_failed(); return Err(err.into());
                    }
                    uploaded = true;
                } else {
                    entry.native.set_scalar_constants(*index, scalars)?;
                }
            }
            if replay { entry.native.launch(command.streams.current().sys)?; }
            else if uploaded { entry.native.record_completion()?; }
            // Only publish the shadow signatures after the entire commit and
            // optional replay succeeded. Failed graphs remain pinned for close.
            for (index, dispatch) in &updates {
                entry.signatures[*index].commit(&dispatch.arguments.info.data);
            }
            Ok(())
        })();
        self.graphs.entries.insert(id, entry);
        result
    }
}
