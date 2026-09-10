use crate::{CannError, sys::*};
use libloading::{Library, Symbol};
use std::ffi::{CStr, OsStr, c_char, c_void};

/// Owns one vendor library. No runtime shutdown, device reset or synchronization
/// is performed on drop. Keep it alive until all foreign resources/work are released.
pub struct CannLibrary(Library);

impl CannLibrary {
    /// # Safety
    /// The path must identify a trusted, ABI-compatible CANN library; its load
    /// and unload hooks must be safe in the caller's process. All work and resources
    /// using its code must be released before dropping the library.
    pub unsafe fn load_from(path: impl AsRef<OsStr>) -> Result<Self, CannError> {
        let path = path.as_ref();
        // SAFETY: the caller establishes the foreign library's load/unload contract.
        unsafe { Library::new(path) }
            .map(Self)
            .map_err(|error| CannError::Library {
                path: path.to_string_lossy().into_owned(),
                message: error.to_string(),
            })
    }

    /// Resolve additional ACL/ACLNN entry points from this library.
    ///
    /// # Safety
    /// T must match the symbol's exact C ABI and signature from the installed SDK.
    /// Callers must obey that function's pointer, lifetime and synchronization rules.
    pub unsafe fn symbol<T>(&self, name: &CStr) -> Result<Symbol<'_, T>, CannError> {
        // SAFETY: signature compatibility is required of the caller; Symbol borrows self.
        unsafe { self.0.get(name.to_bytes_with_nul()) }.map_err(|error| CannError::Symbol {
            name: name.to_string_lossy().into_owned(),
            message: error.to_string(),
        })
    }
}

macro_rules! acl_api {
    ($(fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $result:ty;)+) => {
        #[allow(non_snake_case)]
        pub struct CannApi {
            library: CannLibrary,
            $($name: unsafe extern "C" fn($($ty),*) -> $result,)+
        }

        #[allow(non_snake_case)]
        impl CannApi {
            /// Load the platform's AscendCL library using the OS loader search path.
            ///
            /// # Safety
            /// Same requirements as load_from; the search path must resolve to a
            /// trusted CANN library. This does not call aclInit or select a device.
            pub unsafe fn load() -> Result<Self, CannError> {
                let filename = libloading::library_filename("ascendcl");
                // SAFETY: the caller guarantees the loader search path and library contract.
                unsafe { Self::load_from(filename) }
            }

            /// Resolve the runtime API without calling it. Missing symbols are errors.
            ///
            /// # Safety
            /// The library must implement the AscendCL ABI for every declared method.
            /// All ACL objects and pending work must be released before this API is dropped.
            pub unsafe fn load_from(path: impl AsRef<OsStr>) -> Result<Self, CannError> {
                // SAFETY: the caller guarantees library ABI and load/unload compatibility.
                let library = unsafe { CannLibrary::load_from(path)? };
                Ok(Self {
                    $(
                        // SAFETY: signatures are fixed to the AscendCL API. The owning
                        // library stays in Self; copied pointers are never exposed as safe calls.
                        $name: unsafe { *library.symbol::<unsafe extern "C" fn($($ty),*) -> $result>(
                            CStr::from_bytes_with_nul(concat!(stringify!($name), "\0").as_bytes()).unwrap()
                        )? },
                    )+
                    library,
                })
            }

            pub fn library(&self) -> &CannLibrary { &self.library }

            $(
                /// # Safety
                /// Follow this AscendCL function's SDK initialization and thread-local
                /// context requirements. Pointers,
                /// sizes and handles must be valid; asynchronous inputs/outputs must
                /// outlive completion. Release/reset/finalize only after affected work
                /// is synchronized and no other caller can access those resources.
                pub unsafe fn $name(&self, $($arg: $ty),*) -> $result {
                    // SAFETY: all foreign-call preconditions are required of the caller.
                    unsafe { (self.$name)($($arg),*) }
                }
            )+
        }
    };
}

acl_api! {
    fn aclInit(config_path: *const c_char) -> AclError;
    fn aclFinalize() -> AclError;
    fn aclGetRecentErrMsg() -> *const c_char;
    fn aclrtGetDeviceCount(count: *mut u32) -> AclError;
    fn aclrtSetDevice(device_id: i32) -> AclError;
    fn aclrtGetDevice(device_id: *mut i32) -> AclError;
    fn aclrtResetDevice(device_id: i32) -> AclError;
    fn aclrtCreateContext(context: *mut AclContext, device_id: i32) -> AclError;
    fn aclrtDestroyContext(context: AclContext) -> AclError;
    fn aclrtSetCurrentContext(context: AclContext) -> AclError;
    fn aclrtGetCurrentContext(context: *mut AclContext) -> AclError;
    fn aclrtCreateStream(stream: *mut AclStream) -> AclError;
    fn aclrtDestroyStream(stream: AclStream) -> AclError;
    fn aclrtSynchronizeStream(stream: AclStream) -> AclError;
    fn aclrtCreateEvent(event: *mut AclEvent) -> AclError;
    fn aclrtDestroyEvent(event: AclEvent) -> AclError;
    fn aclrtRecordEvent(event: AclEvent, stream: AclStream) -> AclError;
    fn aclrtQueryEventStatus(event: AclEvent, status: *mut AclEventRecordedStatus) -> AclError;
    fn aclrtSynchronizeEvent(event: AclEvent) -> AclError;
    fn aclrtStreamWaitEvent(stream: AclStream, event: AclEvent) -> AclError;
    fn aclrtMalloc(device_ptr: *mut *mut c_void, size: usize, policy: AclMemMallocPolicy) -> AclError;
    fn aclrtFree(device_ptr: *mut c_void) -> AclError;
    fn aclrtMallocHost(host_ptr: *mut *mut c_void, size: usize) -> AclError;
    fn aclrtFreeHost(host_ptr: *mut c_void) -> AclError;
    fn aclrtMemcpy(dst: *mut c_void, dest_max: usize, src: *const c_void, count: usize, kind: AclMemcpyKind) -> AclError;
    fn aclrtMemcpyAsync(dst: *mut c_void, dest_max: usize, src: *const c_void, count: usize, kind: AclMemcpyKind, stream: AclStream) -> AclError;
    fn aclrtGetMemInfo(attr: AclMemAttr, free: *mut usize, total: *mut usize) -> AclError;
}
