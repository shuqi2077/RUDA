//! Explicit, synchronous Ascend950DT BF16 kernels, adapted from the supplied
//! DeepGEMM-Ascend device headers. No PyTorch, CUDA, DeepJIT or CPU fallback.
//! This does not implement Ruda's generic Kernel IR compiler/Runtime trait.
mod artifact;
pub(super) mod native;
mod plan;
pub use plan::{GemmKind,GemmSpec,GroupEnds,Transpose};
use super::{CannSession,CannTensor,DType,layout::invalid};
use crate::{CannError,check_status};
use native::{Arguments,GmPtr,Epilogue,Kernel,NativeApi};
use std::{cell::{Cell,RefCell},collections::HashMap,path::{Path,PathBuf},rc::Rc};

#[derive(Debug,Clone,Copy,Default,PartialEq,Eq)]
pub struct KernelStats {pub modules_loaded:u64,pub launches:u64,pub synchronizations:u64}

/// A cached native-kernel provider tied to one borrowed CANN session/context/stream.
/// Not Send/Sync. Every compute call completes its stream before returning.
/// Pending/failed work never permits tensors or executable code to be freed early.
pub struct DeepGemm {
    api:Rc<NativeApi>,root:PathBuf,kernels:RefCell<HashMap<String,Kernel>>,
    stats:Cell<KernelStats>,poisoned:Cell<bool>,
}
impl DeepGemm {
    /// # Safety
    /// Artifact directory must contain trusted, unmodified outputs of build_deepgemm.py
    /// from this release. Executable data cannot be made safe by shape/hash checks.
    /// Session attach() ownership obligations continue to apply.
    pub unsafe fn load(session:&Rc<CannSession>,artifact_root:impl AsRef<Path>)->Result<Self,CannError> {
        if cfg!(not(target_pointer_width="64")) || cfg!(not(target_endian="little")) {return Err(invalid("DeepGEMM kernel ABI requires a 64-bit little-endian host"));}
        let root=std::fs::canonicalize(artifact_root).map_err(|e|invalid(e.to_string()))?;
        let api=NativeApi::resolve(session)?;api.check_soc()?;
        Ok(Self{api,root,kernels:RefCell::new(HashMap::new()),stats:Cell::new(KernelStats::default()),poisoned:Cell::new(false)})
    }
    pub fn stats(&self)->KernelStats {self.stats.get()}
    pub fn session(&self)->&Rc<CannSession> {&self.api.session}
    pub fn prepare(&self,spec:&GemmSpec)->Result<(),CannError> {
        if self.poisoned.get(){return Err(invalid("DeepGEMM provider poisoned by a failed execution; close and recreate after successful synchronization"));}
        self.api.session.bind()?;
        let key=spec.key();
        if !self.kernels.borrow().contains_key(&key) {
            let (image,name)=artifact::read(&self.root,&key)?;
            let kernel=Kernel::load(self.api.clone(),image,&name)?;
            self.kernels.borrow_mut().insert(key,kernel);
            let mut st=self.stats.get();st.modules_loaded+=1;self.stats.set(st);
        }
        Ok(())
    }
    /// New output allocation. Use gemm_into() to reuse output storage between calls.
    pub fn gemm(&self,a:&CannTensor,b:&CannTensor,ta:Transpose,tb:Transpose,dtype:DType)->Result<CannTensor,CannError> {
        let spec=GemmSpec::new(a.layout(),b.layout(),ta,tb,dtype)?;
        self.api.session.same_session(&[a,b])?;
        self.prepare(&spec)?;
        let mut out=self.api.session.allocate_tensor(spec.out.shape(),dtype)?;
        self.run(&spec,a,b,&mut out,None)?;Ok(out)
    }
    pub fn gemm_into(&self,a:&CannTensor,b:&CannTensor,ta:Transpose,tb:Transpose,out:&mut CannTensor)->Result<(),CannError> {
        let spec=GemmSpec::new(a.layout(),b.layout(),ta,tb,out.layout().dtype())?;
        self.run(&spec,a,b,out,None)
    }
    /// Upload checked, immutable physical row ends once; no CPU reads of device indices.
    pub fn upload_group_ends(&self,ends:GroupEnds)->Result<DeviceGroupEnds,CannError> {
        let bytes:Vec<u8>=ends.ends().iter().flat_map(|x|x.to_ne_bytes()).collect();
        let data=self.api.session.from_bytes(&[ends.ends().len() as i64],DType::I32,&bytes)?;
        Ok(DeviceGroupEnds{ends,data})
    }
    pub fn grouped_nt(&self,a:&CannTensor,weights:&CannTensor,groups:&DeviceGroupEnds,dtype:DType)->Result<CannTensor,CannError> {
        let spec=GemmSpec::grouped_nt(a.layout(),weights.layout(),groups.ends.clone(),dtype)?;
        self.api.session.same_session(&[a,weights,&groups.data])?;self.prepare(&spec)?;
        let mut out=self.api.session.allocate_tensor(spec.out.shape(),dtype)?;
        self.run(&spec,a,weights,&mut out,Some(&groups.data))?;Ok(out)
    }
    /// Reuse an existing [M,N] output allocation for the aligned expert kernel.
    pub fn grouped_nt_into(&self,a:&CannTensor,weights:&CannTensor,groups:&DeviceGroupEnds,out:&mut CannTensor)->Result<(),CannError> {
        let spec=GemmSpec::grouped_nt(a.layout(),weights.layout(),groups.ends.clone(),out.layout().dtype())?;
        self.run(&spec,a,weights,out,Some(&groups.data))
    }
    /// Dense linear backward for Y = X W^T. Returns dX (BF16) and dW (F32).
    /// Requires BF16 dY and rank-2 tensors. No automatic autograd registration.
    /// Two native GEMMs; no complete transposed copies or FP32 input expansions.
    pub fn linear_nt_backward(&self,x:&CannTensor,w:&CannTensor,dy:&CannTensor)->Result<(CannTensor,CannTensor),CannError> {
        let forward=GemmSpec::new(x.layout(),w.layout(),Transpose::No,Transpose::Yes,DType::BF16)?;
        if forward.kind!=GemmKind::Dense || dy.layout()!=forward.output_layout(){return Err(invalid("linear backward requires dense BF16 dY matching X @ W^T"));}
        self.api.session.same_session(&[x,w,dy])?;
        let dx_spec=GemmSpec::new(dy.layout(),w.layout(),Transpose::No,Transpose::No,DType::BF16)?;
        let dw_spec=GemmSpec::new(dy.layout(),x.layout(),Transpose::Yes,Transpose::No,DType::F32)?;
        self.prepare(&dx_spec)?;self.prepare(&dw_spec)?;
        let dx=self.gemm(dy,w,Transpose::No,Transpose::No,DType::BF16)?;
        let dw=self.gemm(dy,x,Transpose::Yes,Transpose::No,DType::F32)?;
        Ok((dx,dw))
    }
    fn run(&self,s:&GemmSpec,a:&CannTensor,b:&CannTensor,out:&mut CannTensor,groups:Option<&CannTensor>)->Result<(),CannError> {
        self.api.session.same_session(&[a,b,out])?;
        if a.layout()!=&s.a||b.layout()!=&s.b||out.layout()!=&s.out{return Err(invalid("native GEMM tensor layout mismatch"));}
        if let Some(g)=groups{self.api.session.same_session(&[g])?;}
        // Inputs and output are separately owned, non-cloneable CannTensors. Also check
        // raw allocation identity to catch broken foreign/mock allocators before launch.
        let aa=a.buffer.data.as_ptr();let bb=b.buffer.data.as_ptr();let dd=out.buffer.data.as_ptr();
        if aa==dd||bb==dd{return Err(invalid("GEMM output aliases input"));}
        self.prepare(s)?;
        let batched=s.kind==GemmKind::Batched;
        let mut args=Arguments{
            a:GmPtr{addr:aa as u64,stride_outer:*s.a.shape().last().unwrap() as u64,stride_batch:if batched{s.m as u64*s.k as u64}else{0}},
            b:GmPtr{addr:bb as u64,stride_outer:*s.b.shape().last().unwrap() as u64,stride_batch:if batched{s.n as u64*s.k as u64}else{0}},
            d:GmPtr{addr:dd as u64,stride_outer:s.n as u64,stride_batch:if batched{s.m as u64*s.n as u64}else{0}},
            m:s.m,n:s.n,k:s.k,groups_ptr:groups.map_or(std::ptr::null_mut(),|t|t.buffer.data.as_ptr()),
            groups:if s.kind==GemmKind::Dense{0}else{s.groups},
            epilogue:Epilogue{alpha:1.0,pad0:0,sfd:0,stride:0,n:s.n,pad1:0},
        };
        let mut pointers=args.pointers();
        let kernels=self.kernels.borrow();let kernel=kernels.get(&s.key()).unwrap();
        self.api.session.quiescent.set(false);
        // SAFETY: verified kernel contract has nine arguments; all host arguments,
        // allocations, module code and metadata survive through stream completion.
        let launch=unsafe{(self.api.launch)(kernel.function.as_ptr(),32,self.api.session.stream,std::ptr::null_mut(),pointers.as_mut_ptr())};
        let sync=unsafe{self.api.session.api.aclrtSynchronizeStream(self.api.session.stream)};
        self.api.session.quiescent.set(sync==0);
        let mut st=self.stats.get();st.launches+=1;st.synchronizations+=1;self.stats.set(st);
        if launch!=0||sync!=0 {self.poisoned.set(true);}
        if sync!=0{return Err(CannError::Completion{operation:"DeepGEMM Ascend GEMM",launch_code:launch,sync_code:sync});}
        check_status("aclrtLaunchKernelWithArgsArray",launch)
    }
}

/// Kept private to prevent device-side prefix sums changing after validation.
pub struct DeviceGroupEnds {ends:GroupEnds,data:CannTensor}
impl DeviceGroupEnds {pub fn host_ends(&self)->&[i32]{self.ends.ends()}}
