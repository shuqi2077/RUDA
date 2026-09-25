//! Explicit fixed-address kernel graph construction and native driver replay.
//!
//! This is NOT arbitrary PyTorch/CUDA stream capture: only prepared RUDA
//! kernels are accepted. There are no host callbacks, transfers, allocations,
//! dynamic grids, tensor maps or cross-device nodes in the graph.
//! Shapes/addresses are fixed. Runtime scalars can be updated with a matching prepared node.
use crate::{CudaRuntime, CudaCompiler, execution::CudaServer};
use ruda::runtime::{client::ComputeClient, compiler::RudaTask,
    server::{KernelArguments, RudaCount, ServerError}};
use ruda_core::device_handle::DeviceHandle;
use ruda_kernel::dsl::prelude::PreparedKernel;
use crate::execution::graph::error;
use crate::execution::graph_topology::{GraphTopology, to_dot};

pub(crate) struct GraphDispatch {
    pub task: Box<dyn RudaTask<CudaCompiler>>,
    pub count: RudaCount,
    pub arguments: KernelArguments,
    pub scalar_words: usize,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum GraphCommand { Replay, Synchronize, Query, QueryCompletion, WaitCompletion, TryClose, Close }

pub(crate) fn validate_dispatch(count: &RudaCount, args: &KernelArguments) -> Result<(), ServerError> {
    let RudaCount::Static(x, y, z) = count else {
        return Err(error("native graph rejects device-read/dynamic dispatch dimensions"));
    };
    if *x == 0 || *y == 0 || *z == 0 || *x > i32::MAX as u32 || *y > 65535 || *z > 65535 {
        return Err(error("invalid static native graph grid dimensions"));
    }
    if !args.tensor_maps.is_empty() {
        return Err(error("native graph tensor-map/TMA nodes are not supported in this revision"));
    }
    if args.info.dynamic_metadata_offset > args.info.data.len() {
        return Err(error("invalid graph scalar/metadata boundary"));
    }
    Ok(())
}

/// Owns an instantiated native graph on a fixed RUDA device/queue.
///
/// Graph operations are serialized by the same device service as normal kernels.
/// The object retains the client, buffer/view handles and compiled module owner.
/// Replay does not wait; `synchronize`/`close` do. Graph construction and destruction
/// are not intended for the token-by-token hot path.
pub struct CudaGraph {
    client: ComputeClient<CudaRuntime>,
    id: Option<u64>,
    nodes: usize,
    dependencies: Vec<Vec<usize>>,
}
impl CudaGraph {
    /// Build an ordered chain of explicit prepared kernels. No graph kernels
    /// execute here; compilation, metadata copies and graph upload can occur.
    ///
    /// # Safety
    /// Every argument must belong to this client/device and match its kernel,
    /// with sufficient live storage and valid aliasing. Kernel bounds checks do
    /// not verify data types, absence of races, or semantic shapes. Prepared
    /// kernels must not use hidden pointers, textures, or dynamic parallelism.
    /// Only retained explicit buffer arguments may be accessed. External users
    /// must synchronize any access not submitted through this RUDA client.
    pub unsafe fn build(client: &ComputeClient<CudaRuntime>, nodes: Vec<PreparedKernel<CudaRuntime>>)
        -> Result<Self, ServerError>
    {
        // Default remains untracked: no extra completion event in the replay hot path.
        unsafe { Self::build_impl(client, nodes, false, None) }
    }

    /// Build with an owned completion event. Each replay records one event after
    /// launch; `query_completion`/`wait_completion` ignore work submitted later.
    /// This adds event-record overhead, so it is opt-in, not a promised speedup.
    ///
    /// # Safety
    /// The same device, buffer lifetime and kernel semantics contract as `build`.
    pub unsafe fn build_tracked(client: &ComputeClient<CudaRuntime>, nodes: Vec<PreparedKernel<CudaRuntime>>)
        -> Result<Self, ServerError>
    {
        unsafe { Self::build_impl(client, nodes, true, None) }
    }

    /// Build a fixed-address dependency graph rather than forcing a chain.
    /// `dependencies[i]` lists the parents of node i (all must be < i). Empty
    /// parent lists are independent roots. Shared read-only inputs are allowed;
    /// unordered overlapping writable views are rejected using kernel IR access
    /// declarations. Source-only tasks without inspectable IR are rejected.
    /// This is not stream capture and does not guarantee concurrent execution.
    ///
    /// # Safety
    /// All `build` requirements apply. IR visibility and explicit bindings must
    /// faithfully describe every access. Hidden pointers, cross-node signalling,
    /// and races within individual kernels cannot be validated by this API.
    pub unsafe fn build_dag(client: &ComputeClient<CudaRuntime>, nodes: Vec<PreparedKernel<CudaRuntime>>,
        dependencies: Vec<Vec<usize>>) -> Result<Self, ServerError>
    {
        unsafe { Self::build_impl(client, nodes, false, Some(dependencies)) }
    }

    /// Dependency graph with the same opt-in completion event as `build_tracked`.
    ///
    /// # Safety
    /// Same requirements as `build_dag`.
    pub unsafe fn build_dag_tracked(client: &ComputeClient<CudaRuntime>, nodes: Vec<PreparedKernel<CudaRuntime>>,
        dependencies: Vec<Vec<usize>>) -> Result<Self, ServerError>
    {
        unsafe { Self::build_impl(client, nodes, true, Some(dependencies)) }
    }

    /// Infer memory dependencies from a sequence of prepared kernels. Every
    /// conflicting read/write pair preserves its order in `nodes`; independent
    /// branches remain unordered. Exact byte views and kernel IR access modes
    /// are used, and redundant transitive edges are omitted. No kernels are
    /// executed while planning. Existing ordered/explicit-DAG APIs are unchanged.
    ///
    /// # Safety
    /// All `build_dag` requirements apply. The ORIGINAL order must have correct
    /// sequential semantics. All cross-kernel effects must be described by the
    /// explicit buffers and their real IR visibility. Hidden pointers, counters
    /// outside those buffers, external I/O and timing-dependent signalling are
    /// not supported. Use explicit dependencies for non-memory ordering.
    pub unsafe fn build_inferred(client: &ComputeClient<CudaRuntime>,
        nodes: Vec<PreparedKernel<CudaRuntime>>) -> Result<Self, ServerError>
    {
        unsafe { Self::build_inferred_impl(client, nodes, false) }
    }

    /// Dependency inference with the same opt-in completion event as
    /// `build_tracked`. Not an automatic PyTorch model capture interface.
    ///
    /// # Safety
    /// Same requirements as `build_inferred`.
    pub unsafe fn build_inferred_tracked(client: &ComputeClient<CudaRuntime>,
        nodes: Vec<PreparedKernel<CudaRuntime>>) -> Result<Self, ServerError>
    {
        unsafe { Self::build_inferred_impl(client, nodes, true) }
    }

    unsafe fn build_inferred_impl(client: &ComputeClient<CudaRuntime>,
        nodes: Vec<PreparedKernel<CudaRuntime>>, track_completion: bool) -> Result<Self, ServerError>
    {
        if nodes.is_empty() || nodes.len() > 4096 { return Err(error("graph requires 1..=4096 nodes")); }
        let client = client.fixed_execution_queue();
        let mut dispatches = Vec::with_capacity(nodes.len());
        for node in nodes {
            let scalar_words = node.scalar_words();
            let (task, count, arguments, origin) = node.into_parts();
            if !client.same_execution_queue(&origin) {
                return Err(error("graph nodes must use the same device and fixed execution queue"));
            }
            validate_dispatch(&count, &arguments)?;
            dispatches.push(GraphDispatch { task, count, arguments, scalar_words });
        }
        let stream = client.execution_stream();
        let (id, nodes, dependencies) = DeviceHandle::<CudaServer>::new(client.device_id())
            .submit_blocking(move |server| server.graph_build_inferred(stream, dispatches, track_completion))
            .map_err(|e| error(format!("graph inference device queue: {e:?}")))??;
        Ok(Self { client, id: Some(id), nodes, dependencies })
    }

    unsafe fn build_impl(client: &ComputeClient<CudaRuntime>, nodes: Vec<PreparedKernel<CudaRuntime>>,
        track_completion: bool, dependencies: Option<Vec<Vec<usize>>>) -> Result<Self, ServerError>
    {
        if nodes.is_empty() || nodes.len() > 4096 { return Err(error("graph requires 1..=4096 nodes")); }
        let check_accesses = dependencies.is_some();
        // Reject malformed topology before device-service submission or compilation.
        let topology = match dependencies {
            Some(parents) => GraphTopology::new(nodes.len(), parents),
            None => GraphTopology::chain(nodes.len()),
        }.map_err(|e| error(format!("graph topology: {e:?}")))?;
        let dependencies = topology.parents().to_vec();
        let client = client.fixed_execution_queue();
        let mut dispatches = Vec::with_capacity(nodes.len());
        for node in nodes {
            let scalar_words = node.scalar_words();
            let (task, count, arguments, origin) = node.into_parts();
            if !client.same_execution_queue(&origin) {
                return Err(error("graph nodes must use the same device and fixed execution queue"));
            }
            validate_dispatch(&count, &arguments)?;
            dispatches.push(GraphDispatch { task, count, arguments, scalar_words });
        }
        let stream = client.execution_stream();
        let (id, nodes) = DeviceHandle::<CudaServer>::new(client.device_id())
            .submit_blocking(move |server| server.graph_build(stream, dispatches, track_completion, topology, check_accesses))
            .map_err(|e| error(format!("graph build device queue: {e:?}")))??;
        Ok(Self { client, id: Some(id), nodes, dependencies })
    }
    /// Number of native kernel nodes. Replay submits ONE driver graph launch.
    pub fn node_count(&self) -> usize { self.nodes }
    /// Fixed dependency edges, in the original node numbering. Scalar updates
    /// and replays never replace this topology.
    pub fn dependencies(&self) -> &[Vec<usize>] { &self.dependencies }
    pub fn edge_count(&self) -> usize { self.dependencies.iter().map(Vec::len).sum() }
    /// Export the recorded structure without addresses, handles or scalar values.
    /// This is a host description, not a driver profiler or a timing trace.
    pub fn to_dot(&self) -> String { to_dot(&self.dependencies) }
    /// Queue a replay against the same live device addresses. Does not compile,
    /// upload metadata, allocate tensor storage, or loop over kernel launches.
    pub fn replay(&mut self) -> Result<(), ServerError> { self.command(GraphCommand::Replay).map(|_| ()) }
    /// Wait for the fixed queue; includes preceding non-graph work on that queue.
    pub fn synchronize(&mut self) -> Result<(), ServerError> { self.command(GraphCommand::Synchronize).map(|_| ()) }
    /// Query completion of the entire fixed execution queue without waiting for
    /// the GPU. This can be false because of later non-graph work on that queue.
    pub fn query(&mut self) -> Result<bool, ServerError> { self.command(GraphCommand::Query) }

    /// Query the latest graph submission (or graph-owned metadata upload), not
    /// the entire stream tail. Requires `build_tracked`; does not implicitly
    /// allocate an event or synchronize. This is NOT a per-replay ticket.
    pub fn query_completion(&mut self) -> Result<bool, ServerError> {
        self.command(GraphCommand::QueryCompletion)
    }

    /// Wait on the owned completion marker, excluding later unrelated work.
    /// Existing synchronize/close/try_close keep their whole-queue semantics.
    pub fn wait_completion(&mut self) -> Result<(), ServerError> {
        self.command(GraphCommand::WaitCompletion).map(|_| ())
    }


    /// Try releasing the graph without a GPU wait. False leaves it fully owned
    /// and replayable; true closes it. Drop/close remain blocking safety fallbacks.
    pub fn try_close(&mut self) -> Result<bool, ServerError> {
        if self.id.is_none() { return Ok(true); }
        if self.command(GraphCommand::TryClose)? { self.id = None; Ok(true) } else { Ok(false) }
    }

    /// Replace only runtime scalar values of one node. No compile, new device
    /// allocation, graph instantiation, shape change or pointer rebinding occurs.
    /// The replacement must be prepared from the identical kernel/specialization,
    /// grid, buffer views and metadata. Node numbering is the original Vec order.
    ///
    /// # Safety
    /// New scalar values must satisfy the kernel's memory, aliasing and control
    /// flow constraints (for example a scalar used as a tensor index). This API
    /// validates structural identity, not the semantic meaning of each scalar.
    /// External accesses must obey the same contract as `build`.
    pub unsafe fn update_node(&mut self, index: usize, node: PreparedKernel<CudaRuntime>)
        -> Result<(), ServerError>
    {
        self.update(index, node, false)
    }

    /// Update one node and queue one replay in the same serialized device-service
    /// operation. Avoids a second host service round trip; not a multi-node transaction.
    ///
    /// # Safety
    /// Same requirements as `update_node`.
    pub unsafe fn update_and_replay(&mut self, index: usize, node: PreparedKernel<CudaRuntime>)
        -> Result<(), ServerError>
    {
        self.update(index, node, true)
    }

    /// Validate and update several scalar-only nodes in ONE device-service call.
    /// All node identities, layouts and scalar destinations are checked first.
    /// An invalid batch changes nothing. Native commit failures poison the graph
    /// and retain resources; this does NOT promise rollback of driver updates.
    /// Duplicates, empty batches and out-of-range indices are rejected.
    ///
    /// # Safety
    /// Every replacement must satisfy `update_node`'s scalar/kernel contract.
    pub unsafe fn update_nodes(&mut self, nodes: Vec<(usize, PreparedKernel<CudaRuntime>)>)
        -> Result<(), ServerError>
    {
        self.update_many(nodes, false)
    }

    /// Update all listed nodes, then replay exactly once, with no intervening
    /// device-service command. Unchanged batches still replay once.
    ///
    /// # Safety
    /// Same requirements and non-rollback failure semantics as `update_nodes`.
    pub unsafe fn update_nodes_and_replay(&mut self, nodes: Vec<(usize, PreparedKernel<CudaRuntime>)>)
        -> Result<(), ServerError>
    {
        self.update_many(nodes, true)
    }

    fn update_many(&self, nodes: Vec<(usize, PreparedKernel<CudaRuntime>)>, replay: bool)
        -> Result<(), ServerError>
    {
        let id = self.id.ok_or_else(|| error("graph is already closed"))?;
        let indices: Vec<_> = nodes.iter().map(|(index, _)| *index).collect();
        crate::execution::graph_batch::check_indices::<()>(self.nodes, &indices)
            .map_err(|e| error(format!("invalid graph update batch: {e:?}")))?;
        let mut dispatches = Vec::with_capacity(nodes.len());
        for (index, node) in nodes {
            let scalar_words = node.scalar_words();
            let (task, count, arguments, origin) = node.into_parts();
            if !self.client.same_execution_queue(&origin) {
                return Err(error("updated graph node belongs to a different device/queue"));
            }
            validate_dispatch(&count, &arguments)?;
            dispatches.push((index, GraphDispatch { task, count, arguments, scalar_words }));
        }
        DeviceHandle::<CudaServer>::new(self.client.device_id())
            .submit_blocking(move |server| server.graph_update_many(id, dispatches, replay))
            .map_err(|e| error(format!("graph batch update device queue: {e:?}")))?
    }

    fn update(&self, index: usize, node: PreparedKernel<CudaRuntime>, replay: bool)
        -> Result<(), ServerError>
    {
        let id = self.id.ok_or_else(|| error("graph is already closed"))?;
        if index >= self.nodes { return Err(error("graph node out of range")); }
        let scalar_words = node.scalar_words();
        let (task, count, arguments, origin) = node.into_parts();
        if !self.client.same_execution_queue(&origin) {
            return Err(error("updated graph node belongs to a different device/queue"));
        }
        validate_dispatch(&count, &arguments)?;
        let dispatch = GraphDispatch { task, count, arguments, scalar_words };
        DeviceHandle::<CudaServer>::new(self.client.device_id())
            .submit_blocking(move |server| server.graph_update(id, index, dispatch, replay))
            .map_err(|e| error(format!("graph update device queue: {e:?}")))?
    }

    /// Wait, destroy the executable/template and release buffer pins. Idempotent.
    /// On failure ownership is retained so cleanup can be retried.
    pub fn close(&mut self) -> Result<(), ServerError> {
        if self.id.is_some() { self.command(GraphCommand::Close)?; self.id = None; }
        Ok(())
    }
    fn command(&self, operation: GraphCommand) -> Result<bool, ServerError> {
        let id = self.id.ok_or_else(|| error("graph is already closed"))?;
        DeviceHandle::<CudaServer>::new(self.client.device_id())
            .submit_blocking(move |server| server.graph_command(id, operation))
            .map_err(|e| error(format!("graph device queue: {e:?}")))?
    }
}
impl Drop for CudaGraph {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            // The server retains the graph and pins for shutdown cleanup.
            log::error!("RUDA graph close failed; resources retained by device service: {e:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn graph_grid_rejects_zero_and_overflow() {
        for grid in [RudaCount::Static(0,1,1), RudaCount::Static(1,0,1),
            RudaCount::Static(1,1,0), RudaCount::Static(u32::MAX,1,1),
            RudaCount::Static(1,65536,1), RudaCount::Static(1,1,65536)] {
            assert!(validate_dispatch(&grid, &KernelArguments::new()).is_err());
        }
    }
    #[test]
    fn graph_grid_valid_boundaries() {
        for grid in [RudaCount::Static(1,1,1), RudaCount::Static(i32::MAX as u32,65535,65535)] {
            assert!(validate_dispatch(&grid, &KernelArguments::new()).is_ok());
        }
    }
}
