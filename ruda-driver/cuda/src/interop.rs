//! Native stream/event interop on the SAME device service as RUDA kernels.
//! Calls are serialized with kernel submission; event waits remain GPU-side.
use crate::execution::CudaServer;
use ruda::runtime::server::{Handle, ServerError};
use ruda_core::{backtrace::BackTrace, device::DeviceId, device_handle::DeviceHandle};

/// CUDA handles borrowed during a synchronous external-runtime operation.
#[derive(Clone, Copy, Debug)]
pub struct NativeScope {
    /// The RUDA device service's current CUDA context, not an owned handle.
    pub context: usize,
    /// The selected RUDA stream, not an owned handle.
    pub stream: usize,
}

/// Execute an external CUDA runtime on the RUDA device service thread.
/// Pending RUDA work on this stream completes before the callback; work the
/// callback submits to this stream completes before this function returns.
/// No graph, allocator, or default kernel-dispatch policy is changed.
///
/// # Safety
/// The callback must not destroy the borrowed context/stream, change the current
/// context, reenter the RUDA device service, or unwind through a foreign ABI.
/// Work submitted to other streams must be joined to this stream before return.
/// External allocations and modules remain the callback owner's responsibility.
/// RUDA buffer pointers must not be accessed without a separate lifetime and
/// dependency agreement; this interface does not export RUDA allocations.
pub unsafe fn native_scope<T, F>(
    device: DeviceId,
    stream: u64,
    operation: F,
) -> Result<T, ServerError>
where
    T: Send + 'static,
    F: FnOnce(NativeScope) -> Result<T, String> + Send + 'static,
{
    DeviceHandle::<CudaServer>::new(device)
        .submit_blocking(move |server| server.interop_native_scope(stream, operation))
        .map_err(|e| ServerError::Generic {
            reason: format!("external CUDA device queue failed: {e:?}"),
            backtrace: BackTrace::capture(),
        })?
}

/// Enqueue a native operation on the RUDA stream without a host synchronize.
///
/// # Safety
/// The callback obeys `native_scope`'s context/reentrancy rules. Unlike that
/// function it must retain every foreign allocation, event and host argument
/// until its queued work completes. Cross-rank event waits may be submitted
/// here; blocking the host to wait for their producer would deadlock rendezvous.
pub unsafe fn enqueue_native<T, F>(
    device: DeviceId,
    stream: u64,
    operation: F,
) -> Result<T, ServerError>
where
    T: Send + 'static,
    F: FnOnce(NativeScope) -> Result<T, String> + Send + 'static,
{
    DeviceHandle::<CudaServer>::new(device)
        .submit_blocking(move |server| server.interop_enqueue(stream, operation))
        .map_err(|e| ServerError::Generic {
            reason: format!("native enqueue queue failed: {e:?}"),
            backtrace: BackTrace::capture(),
        })?
}

// Peer owners operate on an existing ComputeClient queue, which can have a
// thread-derived logical ID outside the explicitly created native-stream IDs.
// Resolve it through the same stream pool as kernel submission, without
// interpreting it as an interop registry ID.
pub(crate) unsafe fn client_scope<T, F>(
    client: &ruda::runtime::client::ComputeClient<crate::CudaRuntime>,
    operation: F,
) -> Result<T, ServerError>
where
    T: Send + 'static,
    F: FnOnce(NativeScope) -> Result<T, String> + Send + 'static,
{
    let stream = client.execution_stream();
    DeviceHandle::<CudaServer>::new(client.device_id())
        .submit_blocking(move |server| server.interop_execution_scope(stream, operation))
        .map_err(|e| ServerError::Generic {
            reason: format!("native client queue failed: {e:?}"),
            backtrace: BackTrace::capture(),
        })?
}

pub(crate) unsafe fn enqueue_client<T, F>(
    client: &ruda::runtime::client::ComputeClient<crate::CudaRuntime>,
    operation: F,
) -> Result<T, ServerError>
where
    T: Send + 'static,
    F: FnOnce(NativeScope) -> Result<T, String> + Send + 'static,
{
    let stream = client.execution_stream();
    DeviceHandle::<CudaServer>::new(client.device_id())
        .submit_blocking(move |server| server.interop_execution_enqueue(stream, operation))
        .map_err(|e| ServerError::Generic {
            reason: format!("native client enqueue failed: {e:?}"),
            backtrace: BackTrace::capture(),
        })?
}

#[derive(Clone, Copy, Debug)]
pub enum StreamCommand {
    Create,
    Validate(u64),
    Query(u64),
    Synchronize(u64),
    /// Create a native CUDA priority stream, retaining CUDA's range clamping.
    CreatePriority(i32),
    /// Return the actual native signed stream priority as U32 bits.
    Priority(u64),
    /// CUDA (least priority, greatest priority) packed as two signed U32 values.
    PriorityRange,
    Record {
        stream: u64,
        event: u64,
        timing: bool,
    },
    Wait {
        stream: u64,
        event: u64,
    },
    EventQuery(u64),
    EventSynchronize(u64),
    EventDestroy(u64),
    Elapsed {
        start: u64,
        end: u64,
    },
    DeviceSynchronize,
    /// Flush RUDA work queued on the stream and return its `CUstream`, so foreign
    /// kernels launched on it are ordered after that work.
    NativeStream(u64),
}

/// Results are stream/event IDs, bools (0/1), f32 elapsed milliseconds bits, or a `CUstream`.
pub fn command(device: DeviceId, command: StreamCommand) -> Result<u64, ServerError> {
    DeviceHandle::<CudaServer>::new(device)
        .submit_blocking(move |s| s.interop_command(command))
        .map_err(|e| ServerError::Generic {
            reason: format!("interop device queue failed: {e:?}"),
            backtrace: BackTrace::capture(),
        })?
}

/// Retain a native allocation until work already submitted to the stream ends.
pub fn record_allocation(device: DeviceId, stream: u64, handle: Handle) -> Result<(), ServerError> {
    let service = ruda_core::device::ServiceId::of::<CudaServer>(device);
    if handle.service != service {
        return Err(ServerError::ForeignHandle {
            handle: format!("{}", handle.service),
            client: format!("{service}"),
            backtrace: BackTrace::capture(),
        });
    }
    DeviceHandle::<CudaServer>::new(device)
        .submit_blocking(move |s| s.interop_record_allocation(stream, handle))
        .map_err(|e| ServerError::Generic {
            reason: format!("record_stream device queue failed: {e:?}"),
            backtrace: BackTrace::capture(),
        })?
}
