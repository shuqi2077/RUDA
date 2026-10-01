//! CANN 9.x dynamic kernel APIs. Only opaque SDK handles cross this boundary.
use super::super::{CannSession, layout::invalid};
use crate::{CannError, check_status};
use std::{ffi::{c_char,c_void,CStr,CString},rc::Rc,ptr::NonNull};

pub(in crate::tensor) type Handle = *mut c_void;
pub(in crate::tensor) struct NativeApi {
    pub load: unsafe extern "C" fn(*const c_void,usize,*const c_void,*mut Handle)->i32,
    pub get: unsafe extern "C" fn(Handle,*const c_char,*mut Handle)->i32,
    pub unload: unsafe extern "C" fn(Handle)->i32,
    pub launch: unsafe extern "C" fn(Handle,u32,Handle,*mut c_void,*mut *mut c_void)->i32,
    pub soc: unsafe extern "C" fn()->*const c_char,
    pub session: Rc<CannSession>,
}
impl NativeApi {
    pub fn resolve(session:&Rc<CannSession>)->Result<Rc<Self>,CannError> {
        // SAFETY: attach guarantees an ABI-matching trusted CANN library.
        unsafe {
            let lib=session.api.library();
            Ok(Rc::new(Self {
                load:*lib.symbol(c"aclrtBinaryLoadFromData")?,get:*lib.symbol(c"aclrtBinaryGetFunction")?,
                unload:*lib.symbol(c"aclrtBinaryUnLoad")?,launch:*lib.symbol(c"aclrtLaunchKernelWithArgsArray")?,
                soc:*lib.symbol(c"aclrtGetSocName")?,session:session.clone(),
            }))
        }
    }
    pub fn check_soc(&self)->Result<(),CannError> {
        self.session.bind()?;
        // SAFETY: current context is bound and SDK returns a borrowed NUL string.
        let p=unsafe {(self.soc)()};
        if p.is_null() {return Err(CannError::NullHandle("aclrtGetSocName"));}
        let name=unsafe {CStr::from_ptr(p)}.to_string_lossy();
        if name!="Ascend950DT" {return Err(invalid(format!("native CANN kernel target is Ascend950DT, device reports {name}; use explicit ACLNN on other devices")));}
        Ok(())
    }
}

pub(in crate::tensor) struct Kernel {
    pub api:Rc<NativeApi>,
    pub binary:NonNull<c_void>,
    pub function:NonNull<c_void>,
    // Keep the verified ELF bytes alive even if a driver retains their address.
    image:Option<Vec<u8>>,
}
impl Kernel {
    pub fn load(api:Rc<NativeApi>,image:Vec<u8>,name:&str)->Result<Self,CannError> {
        api.session.bind()?;
        let name=CString::new(name).map_err(|_|invalid("kernel name contains NUL"))?;
        let mut binary=std::ptr::null_mut();
        // SAFETY: image is verified; options=NULL requests default loading.
        check_status("aclrtBinaryLoadFromData",unsafe{(api.load)(image.as_ptr().cast(),image.len(),std::ptr::null(),&mut binary)})?;
        let binary=NonNull::new(binary).ok_or(CannError::NullHandle("aclrtBinaryLoadFromData"))?;
        let mut function=std::ptr::null_mut();
        let rc=unsafe{(api.get)(binary.as_ptr(),name.as_ptr(),&mut function)};
        if rc!=0 || function.is_null() {
            // No work has been submitted. Unload on lookup failure; if unload itself
            // fails, conservatively retain the module backing memory and session.
            let unload_rc=unsafe{(api.unload)(binary.as_ptr())};
            if unload_rc!=0 {std::mem::forget(api.clone());std::mem::forget(image);}
            check_status("aclrtBinaryGetFunction",rc)?;
            return Err(CannError::NullHandle("aclrtBinaryGetFunction"));
        }
        Ok(Self{api,binary,function:NonNull::new(function).unwrap(),image:Some(image)})
    }
}
impl Drop for Kernel {
    fn drop(&mut self) {
        if self.api.session.releasable() {
            // SAFETY: bound owning context, all session work known complete.
            if unsafe{(self.api.unload)(self.binary.as_ptr())}==0 {return;}
        }
        // Uncertain completion/unload: leak rather than free live kernel code.
        std::mem::forget(self.api.clone());
        if let Some(image)=self.image.take(){std::mem::forget(image);}
    }
}

// Kernel launch ABI shared with ruda-ascend-kernels::ascend::abi_header.
// Explicit padding is initialized; no vendor C++ implementation is linked.
#[derive(Clone,Copy,Debug)]
#[repr(C)]
pub(super) struct GmPtr {pub addr:u64,pub stride_outer:u64,pub stride_batch:u64}
#[derive(Clone,Copy,Debug)]
#[repr(C)]
pub(super) struct Epilogue {pub alpha:f32,pub pad0:u32,pub sfd:u64,pub stride:u64,pub n:u32,pub pad1:u32}
pub(super) struct Arguments {pub a:GmPtr,pub b:GmPtr,pub d:GmPtr,pub m:u32,pub n:u32,pub k:u32,pub groups_ptr:Handle,pub groups:u32,pub epilogue:Epilogue}
impl Arguments {
    pub fn pointers(&mut self)->[*mut c_void;9] {
        [(&mut self.a as *mut GmPtr).cast(),(&mut self.b as *mut GmPtr).cast(),(&mut self.d as *mut GmPtr).cast(),
         (&mut self.m as *mut u32).cast(),(&mut self.n as *mut u32).cast(),(&mut self.k as *mut u32).cast(),
         (&mut self.groups_ptr as *mut Handle).cast(),(&mut self.groups as *mut u32).cast(),(&mut self.epilogue as *mut Epilogue).cast()]
    }
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn rust_kernel_struct_layout(){use std::mem::{size_of,align_of,offset_of}; assert_eq!(size_of::<usize>(),8);assert_eq!(size_of::<GmPtr>(),24);assert_eq!(align_of::<GmPtr>(),8);assert_eq!(offset_of!(GmPtr,stride_outer),8);assert_eq!(size_of::<Epilogue>(),32);assert_eq!(offset_of!(Epilogue,sfd),8);assert_eq!(offset_of!(Epilogue,n),24);}
}
