//! Production device comparisons. No alternate CPU runtime is accepted.
//! Disabling pruning keeps the same statistics, layouts and arithmetic kernel.
use super::*;
use half::f16;
use ruda_core::tensor::data::TensorData;
use ruda_kernel::tensor::{transfer::from_data, readback::into_data_sync};
use ruda_test_runtime::TestRuntime;
type Tensor = RudaTensor<TestRuntime>;

fn tensor(values: Vec<f32>, shape: impl Into<Shape>, dtype: DType) -> Tensor {
    let shape=shape.into();
    let data=if dtype==DType::F16 {
        TensorData::new(values.into_iter().map(f16::from_f32).collect::<Vec<_>>(),shape)
    } else { TensorData::new(values,shape) };
    from_data(data,&Default::default())
}
fn floats(t: Tensor) -> Vec<f32> {
    let dtype=t.dtype;let data=into_data_sync(t);
    if dtype==DType::F16 {data.to_vec::<f16>().unwrap().into_iter().map(f16::to_f32).collect()}
    else {data.to_vec::<f32>().unwrap()}
}
fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i|(((i*17+seed)%97) as f32-48.)/128.).collect()
}
fn runtime() {
    let name=std::any::type_name::<TestRuntime>();
    assert!(name.contains("CudaRuntime"),"real CUDA test runtime required: {name}");
    assert_eq!(floats(tensor(vec![1.],[1],DType::F32)),vec![1.]);
    println!("RUDA_V34_PRUNING_RUST_GPU_EXECUTED={name}");
}
struct Case {
    plan: DevicePlan<TestRuntime>, workspace: OrderedBackwardWorkspace<TestRuntime>,
    q:Tensor,k:Tensor,v:Tensor,qp:Tensor,kp:Tensor,g:Tensor,
    dq:Tensor,dk:Tensor,dv:Tensor,dqp:Tensor,dkp:Tensor,mla:bool,
}
impl Case {
    fn new(mla:bool, dtype:DType, mode:usize, n:usize, length:usize) -> Self {
        runtime();
        let page_size=16;let logical=length.div_ceil(page_size);let pages=2*logical+2;
        let mut ids=Vec::new();let mut positions=Vec::new();
        for i in 0..n {
            ids.push((i%2) as u32);
            positions.push(match mode {
                1=>(i/6).min(length-1) as u32,
                2=>((i*37)%length) as u32,
                3=>if (i/2)/32%2==0 {(31-(i/2)%32) as u32%17} else {(length-1) as u32},
                _=>(i/2).min(length-1) as u32,
            });
        }
        let first=(0..logical as u32).collect::<Vec<_>>();
        let mut second=(logical as u32..2*logical as u32).collect::<Vec<_>>();
        if logical>=2 {second[1]=0;} // shared page, different logical position
        let host=HostPlan::new(page_size,pages,&[first,second,vec![(2*logical) as u32]],
            &[length as u32,length as u32,16],&ids,&positions).unwrap();
        let heads=4;let kh=if mla {1}else{2};let d=5;let dv=if mla {5}else{7};let pos=3;
        let q=tensor(values(n*heads*d,1),[n,heads,d],dtype);
        let k=tensor(values(pages*page_size*kh*d,3),[pages,page_size,kh,d],dtype);
        let v=if mla {k.clone()}else{tensor(values(pages*page_size*kh*dv,5),[pages,page_size,kh,dv],dtype)};
        let qp=tensor(values(n*heads*pos,7),[n,heads,pos],dtype);
        let kp=tensor(values(pages*page_size*pos,11),[pages,page_size,1,pos],dtype);
        let g=tensor(values(n*heads*dv,13),[n,heads,dv],dtype);
        let out=|like:&Tensor|tensor(vec![f32::NAN;like.meta.num_elements()],like.meta.shape().clone(),dtype);
        let dq=out(&q);let dk=out(&k);let dv=out(&v);let dqp=out(&qp);let dkp=out(&kp);
        let plan=DevicePlan::upload(host,&q);let workspace=OrderedBackwardWorkspace::new(&plan,&q).unwrap();
        Self{plan,workspace,q,k,v,qp,kp,g,dq,dk,dv,dqp,dkp,mla}
    }
    fn run(&mut self, pruning:bool, causal:bool, query_grad:bool) -> Vec<Vec<f32>> {
        self.workspace.set_query_pruning(pruning);
        if self.mla {
            unsafe {self.plan.mla_backward_ordered_into(&self.q,&self.qp,&self.k,&self.kp,&self.g,
                query_grad.then_some(&self.dq),query_grad.then_some(&self.dqp),Some(&self.dk),Some(&self.dkp),
                0.37,causal,&mut self.workspace)}.unwrap();
        } else {
            unsafe {self.plan.attention_backward_ordered_into(&self.q,&self.k,&self.v,&self.g,
                query_grad.then_some(&self.dq),Some(&self.dk),Some(&self.dv),0.37,causal,&mut self.workspace)}.unwrap();
        }
        let mut out=vec![floats(self.dk.clone()),floats(if self.mla {self.dkp.clone()}else{self.dv.clone()})];
        if query_grad {out.push(floats(self.dq.clone()));if self.mla {out.push(floats(self.dqp.clone()));}}
        out
    }
}
fn compare(mla:bool,dtype:DType,mode:usize,causal:bool,queries:usize,query_grad:bool) {
    let mut c=Case::new(mla,dtype,mode,queries,113);
    let reference=c.run(false,causal,query_grad);
    let pruned=c.run(true,causal,query_grad);
    // Real same-device bitwise comparison: the numeric contribution order must
    // remain unchanged, including with unsorted positions and shared pages.
    for (a,b) in reference.iter().zip(&pruned) {
        assert!(a.iter().all(|x|x.is_finite()) && b.iter().all(|x|x.is_finite()));
        assert_eq!(a.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),b.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
    }
    assert_eq!(pruned,c.run(true,causal,query_grad));
    let tail=2*16*c.k.meta.shape()[2]*c.k.meta.shape()[3];
    assert!(pruned[0][pruned[0].len()-tail..].iter().all(|&x|x==0.));
}
#[test] fn v34_monotonic_fp32(){compare(false,DType::F32,0,true,193,true);}
#[test] fn v34_duplicates_fp32(){compare(false,DType::F32,1,true,193,true);}
#[test] fn v34_unsorted_fp32(){compare(false,DType::F32,2,true,193,true);}
#[test] fn v34_block_tail_fp32(){compare(false,DType::F32,3,true,193,true);}
#[test] fn v34_monotonic_fp16(){compare(false,DType::F16,0,true,129,true);}
#[test] fn v34_unsorted_fp16(){compare(false,DType::F16,3,true,129,true);}
#[test] fn v34_noncausal_gqa(){compare(false,DType::F32,3,false,97,true);}
#[test] fn v34_history_only_gqa(){compare(false,DType::F32,3,true,129,false);}
#[test] fn v34_empty_queries(){compare(false,DType::F32,0,true,0,true);}
#[test] fn v34_mla_monotonic(){compare(true,DType::F32,0,true,129,true);}
#[test] fn v34_mla_unsorted(){compare(true,DType::F32,2,true,193,true);}
#[test] fn v34_mla_fp16(){compare(true,DType::F16,3,true,129,true);}
#[test] fn v34_mla_noncausal(){compare(true,DType::F32,3,false,97,true);}
#[test] fn v34_mla_history_only(){compare(true,DType::F32,3,true,129,false);}
#[test] fn v34_immutable_plan_identity_and_equivalence(){
    let c=Case::new(false,DType::F32,0,65,113);
    let cloned=c.plan.clone();assert!(Arc::ptr_eq(&cloned.host,&c.plan.host));
    assert!(c.workspace.matches(&cloned,&c.q));
    let equal=DevicePlan::upload(c.plan.host().clone(),&c.q);
    assert!(!Arc::ptr_eq(&equal.host,&c.plan.host));assert!(c.workspace.matches(&equal,&c.q));
    let mut changed=c.plan.host().clone();changed.words[changed.queries]=1;
    let other=DevicePlan::upload(changed,&c.q);assert!(!c.workspace.matches(&other,&c.q));
}
#[test]
#[ignore="explicit timing after correctness; not ordinary acceptance"]
fn benchmark_query_pruning_v34(){
    use std::time::Instant;
    // Resident inputs, one workspace, fixed output buffers. Both modes use the
    // SAME v34 index. Timings INCLUDE gradient readback and its synchronization.
    for (n,length,mode) in [(32usize,128usize,0usize),(256,256,0),(256,256,3)] {
        let mut c=Case::new(false,DType::F32,mode,n,length);
        let expected=c.run(false,true,false);assert_eq!(expected,c.run(true,true,false));
        for i in 0..4 {for prune in if i%2==0 {[false,true]}else{[true,false]} {c.run(prune,true,false);}}
        for sample in 0..7 {for prune in if sample%2==0 {[false,true]}else{[true,false]} {
            let start=Instant::now();let actual=c.run(prune,true,false);let seconds=start.elapsed().as_secs_f64();
            assert_eq!(actual,expected);
            println!("RUDA_V34_PRUNING_BENCH queries={n} length={length} mode={mode} sample={sample} prune={prune} seconds_with_readback={seconds:.9}");
        }}
    }
}
