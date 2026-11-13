//! CUDA IPC transport primitives for native Rust peer-memory kernels.
//! Memory and event setup are separate from numerical work and graph replay.
use crate::{CudaRuntime, interop};
use cudarc::driver::sys::*;
use ruda::runtime::{client::ComputeClient, server::ServerError};
use ruda_core::{backtrace::BackTrace, device::Device};
use ruda_kernel::dsl::Runtime;
fn error(reason: impl core::fmt::Display) -> ServerError {
    ServerError::Generic {
        reason: reason.to_string(),
        backtrace: BackTrace::capture(),
    }
}
fn status(result: CUresult) -> Result<(), String> {
    if result == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("CUDA IPC: {result:?}"))
    }
}
/// Serialized CUDA IPC handle; contains no tensor values.
#[derive(Clone, Copy, Debug)]
pub struct MemoryTicket {
    pub handle: [u8; 64],
    pub bytes: usize,
}
/// IPC-compatible owning allocation. Unlike a pooled RUDA suballocation its
/// address and whole allocation have the same lifetime/export boundary.
pub struct PeerBuffer {
    client: ComputeClient<CudaRuntime>,
    pointer: u64,
    bytes: usize,
    imported: bool,
}
impl PeerBuffer {
    pub fn allocate(
        client: &ComputeClient<CudaRuntime>,
        bytes: usize,
    ) -> Result<Self, ServerError> {
        if bytes == 0 {
            return Err(error("peer allocation must be nonempty"));
        }
        let pointer = unsafe {
            interop::client_scope(client, move |_| {
                let mut pointer = 0u64;
                status(cuMemAlloc_v2(&mut pointer, bytes))?;
                Ok(pointer)
            })
        }?;
        Ok(Self {
            client: client.fixed_execution_queue(),
            pointer,
            bytes,
            imported: false,
        })
    }
    pub fn export(&self) -> Result<MemoryTicket, ServerError> {
        let pointer = self.pointer;
        let handle = unsafe {
            interop::client_scope(&self.client, move |_| {
                let mut handle = core::mem::MaybeUninit::<CUipcMemHandle>::uninit();
                status(cuIpcGetMemHandle(handle.as_mut_ptr(), pointer))?;
                Ok(core::mem::transmute::<CUipcMemHandle, [u8; 64]>(
                    handle.assume_init(),
                ))
            })
        }?;
        Ok(MemoryTicket {
            handle,
            bytes: self.bytes,
        })
    }
    /// The exporter must retain its allocation until this import is dropped and
    /// all kernels/events referencing it have completed on every participating rank.
    pub unsafe fn import(
        client: &ComputeClient<CudaRuntime>,
        ticket: MemoryTicket,
    ) -> Result<Self, ServerError> {
        let pointer = unsafe {
            interop::client_scope(client, move |_| {
                let mut pointer = 0u64;
                let handle = core::mem::transmute::<[u8; 64], CUipcMemHandle>(ticket.handle);
                status(cuIpcOpenMemHandle_v2(&mut pointer, handle, 1))?;
                Ok(pointer)
            })
        }?;
        Ok(Self {
            client: client.fixed_execution_queue(),
            pointer,
            bytes: ticket.bytes,
            imported: true,
        })
    }
    /// Byte address for an explicitly lifetime-managed `ValuePtr::from_native`.
    pub fn address(&self) -> u64 {
        self.pointer
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn client(&self) -> &ComputeClient<CudaRuntime> {
        &self.client
    }
}
impl Drop for PeerBuffer {
    fn drop(&mut self) {
        let pointer = self.pointer;
        let imported = self.imported;
        let _ = unsafe {
            interop::client_scope(&self.client, move |_| {
                status(if imported {
                    cuIpcCloseMemHandle(pointer)
                } else {
                    cuMemFree_v2(pointer)
                })
            })
        };
    }
}
/// Exportable event used for GPU-side producer/consumer rendezvous. Recording
/// and waiting do not synchronize the host and can be ordered between kernels.
pub struct PeerEvent {
    client: ComputeClient<CudaRuntime>,
    raw: usize,
}
impl PeerEvent {
    pub fn new(client: &ComputeClient<CudaRuntime>) -> Result<Self, ServerError> {
        let raw = unsafe {
            interop::client_scope(client, move |_| {
                let mut event = core::ptr::null_mut();
                status(cuEventCreate(
                    &mut event,
                    CUevent_flags::CU_EVENT_INTERPROCESS as u32
                        | CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
                ))?;
                Ok(event as usize)
            })
        }?;
        Ok(Self {
            client: client.fixed_execution_queue(),
            raw,
        })
    }
    pub fn export(&self) -> Result<[u8; 64], ServerError> {
        let raw = self.raw;
        unsafe {
            interop::client_scope(&self.client, move |_| {
                let mut handle = core::mem::MaybeUninit::<CUipcEventHandle>::uninit();
                status(cuIpcGetEventHandle(handle.as_mut_ptr(), raw as CUevent))?;
                Ok(core::mem::transmute::<CUipcEventHandle, [u8; 64]>(
                    handle.assume_init(),
                ))
            })
        }
    }
    pub unsafe fn import(
        client: &ComputeClient<CudaRuntime>,
        ticket: [u8; 64],
    ) -> Result<Self, ServerError> {
        let raw = unsafe {
            interop::client_scope(client, move |_| {
                let mut event = core::ptr::null_mut();
                let handle = core::mem::transmute::<[u8; 64], CUipcEventHandle>(ticket);
                status(cuIpcOpenEventHandle(&mut event, handle))?;
                Ok(event as usize)
            })
        }?;
        Ok(Self {
            client: client.fixed_execution_queue(),
            raw,
        })
    }
    pub fn record(&self) -> Result<(), ServerError> {
        let raw = self.raw;
        unsafe {
            interop::enqueue_client(&self.client, move |s| {
                status(cuEventRecord(raw as CUevent, s.stream as CUstream))
            })
        }
    }
    /// The producer record must have been submitted before this wait: CUDA
    /// treats an event that has never been recorded as already complete.
    pub fn wait(&self) -> Result<(), ServerError> {
        let raw = self.raw;
        unsafe {
            interop::enqueue_client(&self.client, move |s| {
                status(cuStreamWaitEvent(s.stream as CUstream, raw as CUevent, 0))
            })
        }
    }
    pub fn synchronize(&self) -> Result<(), ServerError> {
        let raw = self.raw;
        unsafe {
            interop::client_scope(&self.client, move |_| {
                status(cuEventSynchronize(raw as CUevent))
            })
        }
    }
}
impl Drop for PeerEvent {
    fn drop(&mut self) {
        let raw = self.raw;
        let _ = unsafe {
            interop::client_scope(&self.client, move |_| {
                status(cuEventDestroy_v2(raw as CUevent))
            })
        };
    }
}
