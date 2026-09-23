//! In-process bridge to the public ruDNN paged kernel. No standalone PTX runtime.
use super::*;
use rudnn::paged_attention::{DevicePlan,HostPlan,SplitWorkspace,MAX_SPLITS};
use std::sync::Mutex;

pub(super) static SPLIT_CALLS: AtomicU64=AtomicU64::new(0);
pub(super) static WORKSPACE_ALLOCS: AtomicU64=AtomicU64::new(0);
pub(super) static WORKSPACE_BYTES: AtomicU64=AtomicU64::new(0);

pub struct NativePlan {
    plan:DevicePlan<CudaRuntime>,
    splits:usize,
    // Calls are enqueued under one lock, always on the plan's ordered queue.
    // No workspace tensor escapes, so reuse cannot race another caller's merge.
    workspace:Mutex<Option<SplitWorkspace<CudaRuntime>>>,
}

/// op=0: validate/upload six-word ABI9 schedule spec; op=1: attention (1 or 2 kernels);
/// op=2: release metadata. The C++ owner guarantees all descriptor lifetimes,
/// disjoint output storage and exactly one destruction of the opaque plan.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_paged(op:u32, plan:*mut *mut NativePlan,
    q:*const Descriptor,k:*const Descriptor,v:*const Descriptor,
    qp:*const Descriptor,kp:*const Descriptor,out:*const Descriptor,
    words:*const u32,word_count:usize,spec:*const u32,scale:f32,causal:bool)->i32
{
    checked(|| {
        assert!(!plan.is_null());
        match op {
            0=>{
                assert!(unsafe{(*plan).is_null()} && !q.is_null() && !spec.is_null());
                let spec=unsafe{std::slice::from_raw_parts(spec,6)};
                assert!(!words.is_null() || word_count==0);
                let words=if word_count==0 { &[] } else {unsafe{std::slice::from_raw_parts(words,word_count)}};
                let host=HostPlan::from_packed(spec[0] as usize,spec[1] as usize,
                    spec[2] as usize,spec[3] as usize,spec[4] as usize,words).expect("invalid paged schedule");
                let like=primitives::tensor(&unsafe{View::read(&*q)});
                let splits=spec[5] as usize;
                assert!((1..=MAX_SPLITS).contains(&splits),"paged split count must be 1..32");
                let native=Box::new(NativePlan{plan:DevicePlan::upload(host,&like),splits,workspace:Mutex::new(None)});
                unsafe{*plan=Box::into_raw(native);}
            }
            1=>{
                assert!(!q.is_null() && !k.is_null() && !v.is_null() && !out.is_null());
                assert!(!unsafe{*plan}.is_null());
                let plan=unsafe{&**plan};
                let q=primitives::tensor(&unsafe{View::read(&*q)});
                let k=primitives::tensor(&unsafe{View::read(&*k)});
                let v=primitives::tensor(&unsafe{View::read(&*v)});
                let out=primitives::tensor(&unsafe{View::read(&*out)});
                assert_eq!(qp.is_null(),kp.is_null());
                let position=if qp.is_null(){None}else{
                    Some((primitives::tensor(&unsafe{View::read(&*qp)}),primitives::tensor(&unsafe{View::read(&*kp)})))
                };
                let position=position.as_ref().map(|(a,b)|(a,b));
                if plan.splits>1 && q.meta.num_elements()!=0 {
                    assert_eq!(q.meta.num_dims(),3,"paged query rank must be 3");
                    assert_eq!(v.meta.num_dims(),4,"paged value cache rank must be 4");
                    let (queries,heads,dv)=(q.meta.shape()[0],q.meta.shape()[1],v.meta.shape()[3]);
                    let mut workspace=plan.workspace.lock().expect("paged workspace lock poisoned");
                    let reuse=workspace.as_ref().map(|w| w.matches(&q,queries,heads,dv,plan.splits)).unwrap_or(false);
                    if !reuse {
                        let next=SplitWorkspace::new(&q,queries,heads,dv,plan.splits).expect("invalid/oversized split workspace");
                        WORKSPACE_ALLOCS.fetch_add(1,Ordering::Relaxed);
                        WORKSPACE_BYTES.fetch_add(next.bytes() as u64,Ordering::Relaxed);
                        *workspace=Some(next);
                    }
                    // SAFETY: C++ allocates/checks a disjoint output; the lock
                    // serializes scratch writes and merge on the same queue.
                    unsafe{plan.plan.attention_into_split(&q,&k,&v,position,&out,scale,causal,workspace.as_mut().unwrap())}
                        .expect("RUDA split paged attention failed");
                    SPLIT_CALLS.fetch_add(1,Ordering::Relaxed);
                    LAUNCHES.fetch_add(2,Ordering::Relaxed);
                } else {
                    // SAFETY: C++ allocates a fresh output and checks overlap.
                    unsafe{plan.plan.attention_into(&q,&k,&v,position,&out,scale,causal)}
                        .expect("RUDA fused paged attention failed");
                    if out.meta.num_elements()!=0 { LAUNCHES.fetch_add(1,Ordering::Relaxed); }
                }
                finish_dispatch(&client());
            }
            2=>{if !unsafe{(*plan).is_null()} {unsafe{drop(Box::from_raw(*plan));*plan=std::ptr::null_mut();}}}
            _=>panic!("unknown paged plan operation"),
        }
    })
}
