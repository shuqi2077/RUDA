//! Native stream/event interop on the SAME device service as RUDA kernels.
//! Calls are serialized with kernel submission; event waits remain GPU-side.
use crate::execution::CudaServer;
use ruda_core::{device::DeviceId,device_handle::DeviceHandle,backtrace::BackTrace};
use ruda::runtime::server::{Handle,ServerError};

#[derive(Clone,Copy,Debug)]
pub enum StreamCommand {
    Create, Validate(u64), Query(u64), Synchronize(u64),
    Record { stream:u64, event:u64, timing:bool },
    Wait { stream:u64, event:u64 }, EventQuery(u64), EventSynchronize(u64),
    EventDestroy(u64), Elapsed { start:u64, end:u64 }, DeviceSynchronize,
}

/// Results are stream/event IDs, bools (0/1), or f32 elapsed milliseconds bits.
pub fn command(device:DeviceId, command:StreamCommand)->Result<u64,ServerError> {
    DeviceHandle::<CudaServer>::new(device).submit_blocking(move |s|s.interop_command(command))
        .map_err(|e|ServerError::Generic { reason:format!("interop device queue failed: {e:?}"),backtrace:BackTrace::capture() })?
}

/// Retain a native allocation until work already submitted to the stream ends.
pub fn record_allocation(device:DeviceId, stream:u64, handle:Handle)->Result<(),ServerError> {
    DeviceHandle::<CudaServer>::new(device).submit_blocking(move |s|s.interop_record_allocation(stream,handle))
        .map_err(|e|ServerError::Generic { reason:format!("record_stream device queue failed: {e:?}"),backtrace:BackTrace::capture() })?
}
