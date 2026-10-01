//! Explicitly exclusive process initialization. For torch_npu interop use attach(),
//! never this owner: reset/finalize are process-wide operations.
use super::*;
use std::{ffi::OsStr,sync::atomic::{AtomicBool,Ordering}};
static EXCLUSIVE:AtomicBool=AtomicBool::new(false);

pub(super) struct Owner {
    api:Rc<CannApi>,device:i32,initialized:bool,selected:bool,
    context:AclContext,stream:AclStream,
}
impl Owner {
    fn release(&mut self)->Result<(),CannError> {
        // SAFETY: only this owner initialized ACL; session/tensor lifetimes retain
        // Owner and its API. Uncertain completion aborts cleanup before release.
        unsafe {
            if !self.context.is_null(){check_status("aclrtSetCurrentContext",self.api.aclrtSetCurrentContext(self.context))?;}
            if !self.stream.is_null(){
                check_status("aclrtSynchronizeStream",self.api.aclrtSynchronizeStream(self.stream))?;
                check_status("aclrtDestroyStream",self.api.aclrtDestroyStream(self.stream))?;
                self.stream=std::ptr::null_mut();
            }
            if !self.context.is_null(){check_status("aclrtDestroyContext",self.api.aclrtDestroyContext(self.context))?;self.context=std::ptr::null_mut();}
            if self.selected{check_status("aclrtResetDevice",self.api.aclrtResetDevice(self.device))?;self.selected=false;}
            if self.initialized{check_status("aclFinalize",self.api.aclFinalize())?;self.initialized=false;}
        }
        Ok(())
    }
}
impl Drop for Owner {
    fn drop(&mut self){
        if let Err(error)=self.release(){
            eprintln!("CANN exclusive-owner cleanup failed; runtime retained: {error}");
            std::mem::forget(self.api.clone()); // Keep library loaded, EXCLUSIVE stays true.
        }else{EXCLUSIVE.store(false,Ordering::Release);}
    }
}
impl CannSession {
    /// Initialize ACL and own one device context and stream. Tensors may outlive
    /// the returned Rc; their references keep the environment alive.
    ///
    /// # Safety
    /// No other component may have initialized ACL or initialize/reset/finalize it
    /// during this environment's lifetime. Libraries must be trusted CANN binaries.
    /// Do NOT call in a process using torch_npu; borrow its resources with attach().
    pub unsafe fn open_exclusive(device:crate::CannDevice,acl_library:impl AsRef<OsStr>,operator_library:impl AsRef<OsStr>)->Result<Rc<Self>,CannError>{
        let api=Rc::new(unsafe{CannApi::load_from(acl_library)?});
        let operators=unsafe{CannLibrary::load_from(operator_library)?};
        if EXCLUSIVE.compare_exchange(false,true,Ordering::AcqRel,Ordering::Acquire).is_err(){return Err(invalid("another exclusive CANN environment exists in this process"));}
        let mut owner=Owner{api:api.clone(),device:device.acl_id(),initialized:false,selected:false,context:std::ptr::null_mut(),stream:std::ptr::null_mut()};
        // SAFETY: the caller promised exclusive ACL ownership; handles live in Owner.
        unsafe {
            check_status("aclInit",api.aclInit(std::ptr::null()))?;owner.initialized=true;
            let mut count=0;check_status("aclrtGetDeviceCount",api.aclrtGetDeviceCount(&mut count))?;
            if device.ordinal()>=count{return Err(invalid(format!("device ordinal {} outside count {count}",device.ordinal())));}
            check_status("aclrtSetDevice",api.aclrtSetDevice(device.acl_id()))?;owner.selected=true;
            check_status("aclrtCreateContext",api.aclrtCreateContext(&mut owner.context,device.acl_id()))?;
            if owner.context.is_null(){return Err(CannError::NullHandle("aclrtCreateContext"));}
            check_status("aclrtSetCurrentContext",api.aclrtSetCurrentContext(owner.context))?;
            check_status("aclrtCreateStream",api.aclrtCreateStream(&mut owner.stream))?;
            if owner.stream.is_null(){return Err(CannError::NullHandle("aclrtCreateStream"));}
            let mut session=CannSession::attach(api,operators,owner.context,owner.stream)?;
            Rc::get_mut(&mut session).expect("newly attached session must be unique").owner=Some(Rc::new(owner));
            Ok(session)
        }
    }
}
