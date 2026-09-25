//! Native driver graphs. Only the serialized device service touches these handles.
//! The builder adds existing RUDA kernels with validated dependency edges, not
//! a host loop disguised as replay. No user stream is placed in capture mode.
use cudarc::driver::sys::*;
use ruda::runtime::server::{Binding, LaunchError, ServerError};
use ruda_core::backtrace::BackTrace;
use std::{collections::HashMap, ffi::c_void, ptr};
use ruda::runtime::{memory_management::ManagedMemoryId, server::MetadataBindingInfo};
use ruda_core::stream_id::StreamId;

pub(crate) type ViewKey = (ManagedMemoryId, StreamId, u64, u64, u64);
pub(crate) fn view_key(binding: &Binding) -> ViewKey {
    (binding.memory.id(), binding.stream, binding.size,
     binding.offset_start.unwrap_or(0), binding.offset_end.unwrap_or(0))
}

#[derive(Debug)]
struct PendingMetadata {
    constants: Option<Vec<u64>>,
    device: Option<Binding>,
}

/// Owns every host value used by cuGraphExecKernelNodeSetParams. No pointer into
/// the device-service scratch or a caller's PreparedKernel survives the call.
#[derive(Debug)]
struct StoredNode {
    node: CUgraphNode,
    function: CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared: u32,
    pointers: Vec<u64>,
    constants: Option<Vec<u64>>,
    device_info: Option<Binding>,
    argument_slots: Vec<*mut c_void>,
}
impl StoredNode {
    fn params(&mut self) -> CUDA_KERNEL_NODE_PARAMS {
        self.argument_slots.clear();
        self.argument_slots.extend(self.pointers.iter_mut().map(|v| v as *mut u64 as *mut c_void));
        if let Some(info) = &mut self.constants {
            self.argument_slots.push(info.as_mut_ptr() as *mut c_void);
        }
        let mut params: CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        params.func = self.function;
        params.gridDimX = self.grid.0; params.gridDimY = self.grid.1; params.gridDimZ = self.grid.2;
        params.blockDimX = self.block.0; params.blockDimY = self.block.1; params.blockDimZ = self.block.2;
        params.sharedMemBytes = self.shared;
        params.kernelParams = if self.argument_slots.is_empty() { ptr::null_mut() }
            else { self.argument_slots.as_mut_ptr() };
        params
    }
}


pub(crate) fn error(reason: impl Into<String>) -> ServerError {
    ServerError::Generic { reason: reason.into(), backtrace: BackTrace::capture() }
}
fn status(value: CUresult, operation: &str) -> Result<(), ServerError> {
    if value == CUresult::CUDA_SUCCESS { Ok(()) }
    else { Err(error(format!("{operation}: {value:?}"))) }
}

#[derive(Debug)]
struct CompletionFence {
    event: CUevent,
    // False before submission and after ANY unrecorded/failed device work.
    // An old completed event must never certify a newer upload/launch.
    recorded: bool,
}

#[derive(Debug)]
pub(crate) struct PinnedBinding {
    pub binding: Binding,
    pub pointer: u64,
}

/// Memory references outlive graph launches; modules belong to CudaContext.
/// Drop must run before CudaContext unloads those modules.
#[derive(Debug)]
pub(crate) struct KernelGraph {
    context: CUcontext,
    stream: CUstream,
    graph: CUgraph,
    executable: CUgraphExec,
    dependencies: Vec<Vec<usize>>,
    pub nodes: usize,
    pub pins: Vec<PinnedBinding>,
    failed: bool,
    pin_indices: HashMap<ViewKey, usize>,
    retained_nodes: Vec<StoredNode>,
    pending_metadata: Option<PendingMetadata>,
    completion: Option<CompletionFence>,
}

impl KernelGraph {
    pub fn new(context: CUcontext, stream: CUstream, dependencies: Vec<Vec<usize>>) -> Result<Self, ServerError> {
        let mut graph = ptr::null_mut();
        status(unsafe { cuGraphCreate(&mut graph, 0) }, "cuGraphCreate")?;
        Ok(Self { context, stream, graph, executable: ptr::null_mut(),
            dependencies, nodes: 0, pins: Vec::new(), failed: false,
            pin_indices: HashMap::new(), retained_nodes: Vec::new(), pending_metadata: None, completion: None })
    }

    pub fn retain(&mut self, binding: Binding, pointer: u64) -> Result<(), ServerError> {
        let key = view_key(&binding);
        if let Some(&index) = self.pin_indices.get(&key) {
            if self.pins[index].pointer != pointer {
                return Err(error("same allocation view moved while constructing graph"));
            }
            return Ok(());
        }
        self.pin_indices.insert(key, self.pins.len());
        self.pins.push(PinnedBinding { binding, pointer });
        Ok(())
    }

    pub fn prepare_metadata(&mut self, info: &MetadataBindingInfo,
        device: Option<Binding>, grid_constants: bool) -> Result<(), ServerError>
    {
        if self.pending_metadata.is_some() || info.dynamic_metadata_offset > info.data.len() {
            return Err(error("invalid/pending graph metadata registration"));
        }
        self.pending_metadata = Some(PendingMetadata {
            constants: grid_constants.then(|| info.data[..info.dynamic_metadata_offset].to_vec()),
            device,
        });
        Ok(())
    }

    pub fn mark_failed(&mut self) { self.failed = true; }

    pub fn check_executable(&self) -> Result<(), ServerError> {
        if self.failed || self.executable.is_null() { Err(error("graph is not executable")) }
        else { Ok(()) }
    }

    /// None means the scalar prefix is passed by value. Some means it lives in
    /// an existing device metadata allocation and must be copied on this queue.
    pub fn scalar_device_binding(&self, index: usize) -> Result<Option<Binding>, ServerError> {
        self.check_executable()?;
        let node = self.retained_nodes.get(index).ok_or_else(|| error("graph node out of range"))?;
        if node.constants.is_some() { Ok(None) }
        else { node.device_info.clone().map(Some).ok_or_else(|| error("node has no scalar storage")) }
    }

    /// Validate native scalar storage during batch preflight, not after an
    /// earlier node has already committed its update.
    pub fn validate_scalar_update(&self, index: usize, scalar_words: usize)
        -> Result<Option<Binding>, ServerError>
    {
        self.check_executable()?;
        let node = self.retained_nodes.get(index).ok_or_else(|| error("graph node out of range"))?;
        if let Some(constants) = &node.constants {
            if scalar_words > constants.len() { return Err(error("scalar prefix exceeds constant metadata")); }
            Ok(None)
        } else {
            self.scalar_device_binding(index)
        }
    }

    /// Change the by-value scalar prefix without rebuilding or changing pointers.
    /// Caller has already checked kernel identity and all metadata outside prefix.
    pub fn set_scalar_constants(&mut self, index: usize, scalars: &[u64]) -> Result<(), ServerError> {
        self.check_executable()?;
        let node = self.retained_nodes.get_mut(index).ok_or_else(|| error("graph node out of range"))?;
        let constants = node.constants.as_mut().ok_or_else(|| error("node uses device metadata"))?;
        if scalars.len() > constants.len() { return Err(error("scalar prefix exceeds constant metadata")); }
        constants[..scalars.len()].copy_from_slice(scalars);
        let params = node.params();
        #[cfg(cuda_12000)]
        let result = unsafe { cuGraphExecKernelNodeSetParams_v2(self.executable, node.node, &params) };
        #[cfg(not(cuda_12000))]
        let result = unsafe { cuGraphExecKernelNodeSetParams(self.executable, node.node, &params) };
        if let Err(err) = status(result, "cuGraphExecKernelNodeSetParams") {
            // The caller must not continue with an ambiguous partially updated graph.
            self.failed = true;
            return Err(err);
        }
        Ok(())
    }

    pub fn add_kernel(
        &mut self, func: CUfunction, grid: (u32, u32, u32), block: (u32, u32, u32),
        shared_bytes: u32, parameters: &mut [*mut c_void],
    ) -> Result<(), LaunchError> {
        self.add_kernel_checked(func, grid, block, shared_bytes, parameters)
            .map_err(|e| LaunchError::Unknown {
                reason: format!("native graph node {}: {e:?}", self.nodes),
                backtrace: BackTrace::capture(),
            })
    }

    fn add_kernel_checked(
        &mut self, func: CUfunction, grid: (u32, u32, u32), block: (u32, u32, u32),
        shared_bytes: u32, parameters: &mut [*mut c_void],
    ) -> Result<(), ServerError> {
        if !self.executable.is_null() || self.failed { return Err(error("graph is not building")); }
        let pending = self.pending_metadata.take().ok_or_else(|| error("missing graph node metadata"))?;
        let pointer_count = parameters.len().checked_sub(usize::from(pending.constants.is_some()))
            .ok_or_else(|| error("graph parameter layout mismatch"))?;
        // The internal caller rejects tensor maps; the other slots are aligned
        // pointers to u64 device addresses owned by LaunchArguments.
        let pointers = parameters[..pointer_count].iter()
            .map(|&p| unsafe { *(p as *const u64) }).collect();
        // The CUDA binding layout/API changed in 12.0. Do not guess an ABI or
        // refer to the removed v1 symbol when compiled against 13.x headers.
        let mut params: CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        params.func = func;
        params.gridDimX = grid.0; params.gridDimY = grid.1; params.gridDimZ = grid.2;
        params.blockDimX = block.0; params.blockDimY = block.1; params.blockDimZ = block.2;
        params.sharedMemBytes = shared_bytes;
        params.kernelParams = if parameters.is_empty() { ptr::null_mut() } else { parameters.as_mut_ptr() };
        let incoming = self.dependencies.get(self.nodes)
            .ok_or_else(|| error("native graph has more kernels than its topology"))?;
        let parent_handles: Vec<CUgraphNode> = incoming.iter().map(|&index| {
            self.retained_nodes.get(index).map(|node| node.node)
                .ok_or_else(|| error("native graph parent is not yet constructed"))
        }).collect::<Result<_, _>>()?;
        let dependencies = if parent_handles.is_empty() { ptr::null() } else { parent_handles.as_ptr() };
        let count = parent_handles.len();
        let mut node = ptr::null_mut();
        // CUDA copies each parameter value now. Host argument vectors/scalars
        // need only live through this call, but device buffers remain pinned.
        #[cfg(cuda_12000)]
        let result = unsafe { cuGraphAddKernelNode_v2(&mut node, self.graph, dependencies, count, &params) };
        #[cfg(not(cuda_12000))]
        let result = unsafe { cuGraphAddKernelNode(&mut node, self.graph, dependencies, count, &params) };
        if let Err(e) = status(result, "cuGraphAddKernelNode") { self.failed = true; return Err(e); }
        self.retained_nodes.push(StoredNode { node, function: func, grid, block, shared: shared_bytes,
            pointers, constants: pending.constants, device_info: pending.device,
            argument_slots: Vec::with_capacity(parameters.len()) });
        self.nodes += 1;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<(), ServerError> {
        if self.nodes == 0 || self.nodes != self.dependencies.len() || self.pending_metadata.is_some()
            || self.failed || !self.executable.is_null() {
            return Err(error("cannot instantiate empty, failed, or already instantiated graph"));
        }
        status(unsafe { cuGraphInstantiateWithFlags(&mut self.executable, self.graph, 0) },
               "cuGraphInstantiateWithFlags")?;
        // Upload ahead of the measured replay loop, on the same queue as all
        // prepared metadata copies. Upload is not graph execution.
        status(unsafe { cuGraphUpload(self.executable, self.stream) }, "cuGraphUpload")
    }

    /// Opt-in at build time. The first record includes graph upload and any
    /// graph-owned metadata copies; never expose an unrecorded "ready" event.
    pub fn enable_completion_tracking(&mut self) -> Result<(), ServerError> {
        self.check_executable()?;
        if self.completion.is_some() { return Err(error("graph completion tracking already enabled")); }
        let mut event = ptr::null_mut();
        status(unsafe { cuEventCreate(&mut event, CUevent_flags::CU_EVENT_DISABLE_TIMING as u32) },
            "graph completion create")?;
        self.completion = Some(CompletionFence { event, recorded: false });
        self.record_completion()
    }

    /// Invalidate the previous marker BEFORE an operation can enqueue work.
    pub fn begin_device_work(&mut self) {
        if let Some(completion) = &mut self.completion { completion.recorded = false; }
    }

    /// A single record after the batch's final upload/launch. Untracked graphs
    /// take a no-op branch and retain their original replay cost.
    pub fn record_completion(&mut self) -> Result<(), ServerError> {
        if let Some(completion) = &mut self.completion {
            completion.recorded = false;
            if let Err(e) = status(unsafe { cuEventRecord(completion.event, self.stream) },
                "graph completion record") {
                self.failed = true;
                return Err(e);
            }
            completion.recorded = true;
        }
        Ok(())
    }

    fn completion_event(&self) -> Result<CUevent, ServerError> {
        self.check_executable()?;
        let completion = self.completion.as_ref().ok_or_else(|| error("use CudaGraph::build_tracked for completion queries"))?;
        if !completion.recorded { return Err(error("graph completion marker does not cover latest work")); }
        Ok(completion.event)
    }

    pub fn query_completion(&mut self) -> Result<bool, ServerError> {
        let event = self.completion_event()?;
        match unsafe { cuEventQuery(event) } {
            CUresult::CUDA_SUCCESS => Ok(true),
            CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            result => { self.failed = true; status(result, "graph completion query")?; unreachable!() }
        }
    }

    pub fn wait_completion(&mut self) -> Result<(), ServerError> {
        let event = self.completion_event()?;
        if let Err(e) = status(unsafe { cuEventSynchronize(event) }, "graph completion wait") {
            self.failed = true;
            return Err(e);
        }
        Ok(())
    }

    pub fn launch(&mut self, stream: CUstream) -> Result<(), ServerError> {
        if self.failed || self.executable.is_null() { return Err(error("graph is not executable")); }
        if stream != self.stream { return Err(error("graph replay moved to a different physical stream")); }
        self.begin_device_work();
        if let Err(e) = status(unsafe { cuGraphLaunch(self.executable, stream) }, "cuGraphLaunch") {
            self.failed = true;
            return Err(e);
        }
        self.record_completion()
    }

    /// Non-waiting GPU query of the WHOLE fixed queue (including unrelated work).
    /// Host device-service dispatch can still take time; this is not lock-free.
    pub fn query(&mut self) -> Result<bool, ServerError> {
        self.check_executable()?;
        match unsafe { cuStreamQuery(self.stream) } {
            CUresult::CUDA_SUCCESS => Ok(true),
            CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            result => { self.failed = true; status(result, "graph stream query")?; unreachable!() }
        }
    }

    /// Return false without waiting for GPU completion. Serialization plus the
    /// public build safety contract prevents new external work between query and destruction.
    pub fn try_close(&mut self) -> Result<bool, ServerError> {
        if !self.query()? { return Ok(false); }
        self.destroy_ready()?;
        Ok(true)
    }

    pub fn synchronize(&mut self) -> Result<(), ServerError> {
        if let Err(e) = status(unsafe { cuStreamSynchronize(self.stream) }, "graph stream synchronize") {
            self.failed = true;
            return Err(e);
        }
        Ok(())
    }

    /// Explicit close reports errors and leaves remaining handles/pins owned.
    /// Only close/destruction waits; replay never inserts a host wait.
    pub fn close(&mut self) -> Result<(), ServerError> {
        if self.graph.is_null() && self.executable.is_null() && self.completion.is_none() { return Ok(()); }
        self.synchronize()?;
        self.destroy_ready()
    }

    fn destroy_ready(&mut self) -> Result<(), ServerError> {
        if !self.executable.is_null() {
            status(unsafe { cuGraphExecDestroy(self.executable) }, "cuGraphExecDestroy")?;
            self.executable = ptr::null_mut();
        }
        if !self.graph.is_null() {
            status(unsafe { cuGraphDestroy(self.graph) }, "cuGraphDestroy")?;
            self.graph = ptr::null_mut();
        }
        if let Some(completion) = &self.completion {
            status(unsafe { cuEventDestroy_v2(completion.event) }, "graph completion destroy")?;
            self.completion = None;
        }
        self.retained_nodes.clear();
        self.dependencies.clear();
        self.pending_metadata = None;
        self.pin_indices.clear();
        self.pins.clear();
        Ok(())
    }
}

impl Drop for KernelGraph {
    fn drop(&mut self) {
        if self.graph.is_null() && self.executable.is_null() && self.completion.is_none() { return; }
        let cleanup = (|| {
            let mut previous = ptr::null_mut();
            status(unsafe { cuCtxGetCurrent(&mut previous) }, "graph shutdown get context")?;
            status(unsafe { cuCtxSetCurrent(self.context) }, "graph shutdown set context")?;
            let result = self.close();
            let restore = status(unsafe { cuCtxSetCurrent(previous) }, "graph shutdown restore context");
            result.and(restore)
        })();
        if let Err(e) = cleanup {
            // A failed wait does not prove work stopped. Leaking the pins is
            // safer than freeing storage still reachable from a running graph.
            log::error!("RUDA graph cleanup failed; device allocation pins retained: {e:?}");
            std::mem::forget(std::mem::take(&mut self.pins));
        }
    }
}
