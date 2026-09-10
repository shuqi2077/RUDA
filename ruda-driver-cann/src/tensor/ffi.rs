use crate::{CannError, CannLibrary};
use std::ffi::{CStr, c_void};

#[repr(C)]
pub struct AclTensor {
    _opaque: [u8; 0],
}
#[repr(C)]
pub struct AclScalar {
    _opaque: [u8; 0],
}
#[repr(C)]
pub struct AclIntArray {
    _opaque: [u8; 0],
}
#[repr(C)]
pub struct AclOpExecutor {
    _opaque: [u8; 0],
}
pub type Status = i32;
pub type Run =
    unsafe extern "C" fn(*mut c_void, u64, *mut AclOpExecutor, crate::sys::AclStream) -> Status;
pub type CreateTensor = unsafe extern "C" fn(
    *const i64,
    u64,
    i32,
    *const i64,
    i64,
    i32,
    *const i64,
    u64,
    *mut c_void,
) -> *mut AclTensor;
pub type DestroyTensor = unsafe extern "C" fn(*const AclTensor) -> Status;
pub type ExecutorAction = unsafe extern "C" fn(*mut AclOpExecutor) -> Status;

pub struct OperatorApi {
    pub library: CannLibrary,
    pub create_tensor: CreateTensor,
    pub destroy_tensor: DestroyTensor,
    pub destroy_executor: ExecutorAction,
}

impl OperatorApi {
    pub unsafe fn load(library: CannLibrary) -> Result<Self, CannError> {
        // SAFETY: the caller supplied the matching CANN operator library.
        unsafe {
            Ok(Self {
                create_tensor: *library.symbol(c"aclCreateTensor")?,
                destroy_tensor: *library.symbol(c"aclDestroyTensor")?,
                destroy_executor: *library.symbol(c"aclDestroyAclOpExecutor")?,
                library,
            })
        }
    }
    pub unsafe fn get<T: Copy>(&self, name: &CStr) -> Result<T, CannError> {
        // SAFETY: call sites specify the signature of the named SDK function.
        unsafe { self.library.symbol(name).map(|symbol| *symbol) }
    }
}
