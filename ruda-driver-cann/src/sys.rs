use std::ffi::{c_int, c_void};

pub type AclError = c_int;
pub type AclContext = *mut c_void;
pub type AclStream = *mut c_void;
pub type AclEvent = *mut c_void;
pub type AclMemcpyKind = c_int;
pub type AclMemMallocPolicy = c_int;
pub type AclMemAttr = c_int;
// Integer output preserves unknown values instead of constructing an invalid Rust enum.
pub type AclEventRecordedStatus = c_int;

pub const ACL_SUCCESS: AclError = 0;
pub const ACL_MEMCPY_HOST_TO_HOST: AclMemcpyKind = 0;
pub const ACL_MEMCPY_HOST_TO_DEVICE: AclMemcpyKind = 1;
pub const ACL_MEMCPY_DEVICE_TO_HOST: AclMemcpyKind = 2;
pub const ACL_MEMCPY_DEVICE_TO_DEVICE: AclMemcpyKind = 3;
pub const ACL_MEMCPY_DEFAULT: AclMemcpyKind = 4;
pub const ACL_MEM_MALLOC_HUGE_FIRST: AclMemMallocPolicy = 0;
pub const ACL_MEM_MALLOC_HUGE_ONLY: AclMemMallocPolicy = 1;
pub const ACL_MEM_MALLOC_NORMAL_ONLY: AclMemMallocPolicy = 2;
pub const ACL_DDR_MEM: AclMemAttr = 0;
pub const ACL_HBM_MEM: AclMemAttr = 1;
pub const ACL_EVENT_RECORDED_STATUS_NOT_READY: AclEventRecordedStatus = 0;
pub const ACL_EVENT_RECORDED_STATUS_COMPLETE: AclEventRecordedStatus = 1;
