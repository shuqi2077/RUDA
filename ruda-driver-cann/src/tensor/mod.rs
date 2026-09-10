//! Device-resident, synchronous ACLNN operations. This does not implement Ruda's
//! generic Backend or compile Ruda IR into Ascend kernels.

mod ffi;
mod layout;
mod ops;
use crate::{CannApi, CannError, CannLibrary, check_status, sys::*};
use ffi::*;
use layout::invalid;
pub use layout::{DType, TensorLayout};
pub use ops::ScalarValue;
use std::{cell::Cell, ffi::c_void, ptr::NonNull, rc::Rc};

/// Thread-local borrowed context/stream. Runtime initialization and finalization
/// remain with the caller. All public compute operations synchronize before return.
pub struct CannSession {
    api: Rc<CannApi>,
    ops: OperatorApi,
    context: AclContext,
    stream: AclStream,
    quiescent: Cell<bool>,
}

impl CannSession {
    /// # Safety
    /// ACL must be initialized. context and stream must belong to the same device
    /// and remain valid until this session and all its tensors are dropped. No other
    /// code may enqueue work accessing these tensors or reset/finalize the runtime.
    /// operators must be the trusted matching CANN operator library (libopapi).
    pub unsafe fn attach(
        api: Rc<CannApi>,
        operators: CannLibrary,
        context: AclContext,
        stream: AclStream,
    ) -> Result<Rc<Self>, CannError> {
        if context.is_null() || stream.is_null() {
            return Err(invalid("CANN context and stream must be non-null"));
        }
        // SAFETY: library identity and context/stream ownership are guaranteed by caller.
        let ops = unsafe { OperatorApi::load(operators)? };
        let session = Rc::new(Self {
            api,
            ops,
            context,
            stream,
            quiescent: Cell::new(false),
        });
        session.synchronize()?;
        Ok(session)
    }

    fn bind(&self) -> Result<(), CannError> {
        // SAFETY: attach guarantees the context's validity for the session lifetime.
        check_status("aclrtSetCurrentContext", unsafe {
            self.api.aclrtSetCurrentContext(self.context)
        })
    }

    pub fn synchronize(&self) -> Result<(), CannError> {
        self.bind()?;
        // SAFETY: the stream is valid and exclusively submitted to through this session.
        let code = unsafe { self.api.aclrtSynchronizeStream(self.stream) };
        self.quiescent.set(code == ACL_SUCCESS);
        check_status("aclrtSynchronizeStream", code)
    }

    fn releasable(&self) -> bool {
        self.quiescent.get() && self.bind().is_ok()
    }

    fn allocate(self: &Rc<Self>, bytes: usize) -> Result<Buffer, CannError> {
        self.bind()?;
        let mut data = std::ptr::null_mut();
        // SAFETY: output pointer is initialized; a zero-element tensor still owns
        // a real one-byte allocation, but exposes zero bytes to operations/copies.
        check_status("aclrtMalloc", unsafe {
            self.api
                .aclrtMalloc(&mut data, bytes.max(1), ACL_MEM_MALLOC_NORMAL_ONLY)
        })?;
        let data = NonNull::new(data).ok_or(CannError::NullHandle("aclrtMalloc"))?;
        Ok(Buffer {
            session: self.clone(),
            data,
            bytes,
        })
    }

    fn allocate_tensor(
        self: &Rc<Self>,
        shape: &[i64],
        dtype: DType,
    ) -> Result<CannTensor, CannError> {
        let layout = TensorLayout::contiguous(shape, dtype)?;
        let buffer = self.allocate(layout.byte_len())?;
        // SAFETY: layout is checked contiguous storage; metadata stays with the tensor.
        let descriptor = unsafe {
            (self.ops.create_tensor)(
                layout.shape.as_ptr(),
                layout.shape.len() as u64,
                dtype as i32,
                layout.strides.as_ptr(),
                0,
                2,
                layout.shape.as_ptr(),
                layout.shape.len() as u64,
                buffer.data.as_ptr(),
            )
        };
        let descriptor =
            NonNull::new(descriptor).ok_or(CannError::NullHandle("aclCreateTensor"))?;
        Ok(CannTensor {
            descriptor,
            layout,
            buffer,
        })
    }

    pub fn from_bytes(
        self: &Rc<Self>,
        shape: &[i64],
        dtype: DType,
        bytes: &[u8],
    ) -> Result<CannTensor, CannError> {
        let layout = TensorLayout::contiguous(shape, dtype)?;
        if bytes.len() != layout.byte_len() {
            return Err(invalid("host data length does not match tensor byte count"));
        }
        let tensor = self.allocate_tensor(shape, dtype)?;
        if !bytes.is_empty() {
            // SAFETY: host slice and real device allocation cover the exact synchronous copy.
            check_status("aclrtMemcpy", unsafe {
                self.api.aclrtMemcpy(
                    tensor.buffer.data.as_ptr(),
                    bytes.len(),
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    ACL_MEMCPY_HOST_TO_DEVICE,
                )
            })?;
        }
        Ok(tensor)
    }

    pub fn from_f32(self: &Rc<Self>, shape: &[i64], data: &[f32]) -> Result<CannTensor, CannError> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        self.from_bytes(shape, DType::F32, &bytes)
    }

    pub fn zeros(self: &Rc<Self>, shape: &[i64], dtype: DType) -> Result<CannTensor, CannError> {
        type Memset = unsafe extern "C" fn(*mut c_void, usize, i32, usize) -> AclError;
        let tensor = self.allocate_tensor(shape, dtype)?;
        if tensor.layout.byte_len() != 0 {
            // SAFETY: fixed ACL signature and the tensor owns the full allocation.
            unsafe {
                let memset = self.api.library().symbol::<Memset>(c"aclrtMemset")?;
                check_status(
                    "aclrtMemset",
                    memset(
                        tensor.buffer.data.as_ptr(),
                        tensor.buffer.bytes,
                        0,
                        tensor.buffer.bytes,
                    ),
                )?;
            }
        }
        Ok(tensor)
    }

    fn same_session(self: &Rc<Self>, tensors: &[&CannTensor]) -> Result<(), CannError> {
        if tensors.iter().any(|t| !Rc::ptr_eq(self, &t.buffer.session)) {
            return Err(invalid("tensor belongs to another CANN session"));
        }
        self.bind()
    }

    fn execute(
        self: &Rc<Self>,
        operation: &'static str,
        run: Run,
        prepare: impl FnOnce(*mut u64, *mut *mut AclOpExecutor) -> i32,
    ) -> Result<(), CannError> {
        let mut size = 0u64;
        let mut executor = std::ptr::null_mut();
        check_status(operation, prepare(&mut size, &mut executor))?;
        let executor = NonNull::new(executor).ok_or(CannError::NullHandle(operation))?;
        let mut executor = Executor {
            session: self.clone(),
            handle: Some(executor),
        };
        let size_usize =
            usize::try_from(size).map_err(|_| invalid("workspace size exceeds address space"))?;
        let workspace = if size_usize == 0 {
            None
        } else {
            Some(self.allocate(size_usize)?)
        };
        let pointer = workspace
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.data.as_ptr());
        self.quiescent.set(false);
        // SAFETY: private call sites keep every tensor/scalar alive; workspace
        // remains allocated until successful stream synchronization.
        // The second phase consumes a non-repeatable executor. Only a plan that
        // was never submitted is destroyed by the guard (e.g. allocation failure).
        let handle = executor.handle.take().unwrap();
        let launch_code = unsafe { run(pointer, size, handle.as_ptr(), self.stream) };
        let sync_code = unsafe { self.api.aclrtSynchronizeStream(self.stream) };
        self.quiescent.set(sync_code == ACL_SUCCESS);
        if sync_code != ACL_SUCCESS {
            return Err(CannError::Completion {
                operation,
                launch_code,
                sync_code,
            });
        }
        check_status(operation, launch_code)
    }
}

struct Buffer {
    session: Rc<CannSession>,
    data: NonNull<c_void>,
    bytes: usize,
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if self.session.releasable() {
            // SAFETY: no pending accesses; pointer is uniquely owned and context is bound.
            let _ = unsafe { self.session.api.aclrtFree(self.data.as_ptr()) };
        } else {
            // Failed synchronization is not completion; retain code and allocation.
            std::mem::forget(self.session.clone());
        }
    }
}

struct Executor {
    session: Rc<CannSession>,
    handle: Option<NonNull<AclOpExecutor>>,
}
impl Drop for Executor {
    fn drop(&mut self) {
        if let Some(handle) = self.handle {
            // SAFETY: this executor was returned by phase one but never submitted.
            let _ = unsafe { (self.session.ops.destroy_executor)(handle.as_ptr()) };
        }
    }
}

/// Contiguous device storage. No CPU substitute and no implicit dtype conversion.
pub struct CannTensor {
    descriptor: NonNull<AclTensor>,
    layout: TensorLayout,
    buffer: Buffer,
}
impl CannTensor {
    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }
    pub fn session(&self) -> &Rc<CannSession> {
        &self.buffer.session
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, CannError> {
        self.buffer.session.synchronize()?;
        let mut bytes = vec![0; self.layout.byte_len()];
        if !bytes.is_empty() {
            // SAFETY: owned device allocation and initialized host slice cover the copy.
            check_status("aclrtMemcpy", unsafe {
                self.buffer.session.api.aclrtMemcpy(
                    bytes.as_mut_ptr().cast(),
                    bytes.len(),
                    self.buffer.data.as_ptr(),
                    bytes.len(),
                    ACL_MEMCPY_DEVICE_TO_HOST,
                )
            })?;
        }
        Ok(bytes)
    }
    pub fn to_f32(&self) -> Result<Vec<f32>, CannError> {
        if self.layout.dtype != DType::F32 {
            return Err(invalid("to_f32 requires an F32 tensor"));
        }
        Ok(self
            .to_bytes()?
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    }
}
impl Drop for CannTensor {
    fn drop(&mut self) {
        if self.buffer.session.releasable() {
            // SAFETY: descriptor is uniquely owned; its device storage is freed afterwards.
            let _ = unsafe { (self.buffer.session.ops.destroy_tensor)(self.descriptor.as_ptr()) };
        }
    }
}

#[cfg(test)]
mod tests;
