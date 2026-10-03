//! Real-device compact/full-capacity comparisons. No host or alternate runtime.
//! Inputs, outputs and workspace are resident and reused across both modes.
use super::*;
use half::{bf16, f16};
use ruda_core::tensor::data::TensorData;
use ruda_kernel::tensor::{contiguous::into_contiguous, transfer::from_data, readback::into_data_sync};
use ruda_test_runtime::TestRuntime;
type Tensor = RudaTensor<TestRuntime>;

fn tensor(values: Vec<f32>, shape: impl Into<Shape>, dtype: DType) -> Tensor {
    let shape=shape.into();
    let data=match dtype {
        DType::F16 => TensorData::new(values.into_iter().map(f16::from_f32).collect::<Vec<_>>(),shape),
        DType::BF16 => TensorData::new(values.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),shape),
        _ => TensorData::new(values,shape),
    };
    into_contiguous(from_data(data,&Default::default()))
}
fn floats(t: Tensor) -> Vec<f32> {
    let dtype=t.dtype;let data=into_data_sync(t);
    match dtype {
        DType::F16 => data.to_vec::<f16>().unwrap().into_iter().map(f16::to_f32).collect(),
        DType::BF16 => data.to_vec::<bf16>().unwrap().into_iter().map(bf16::to_f32).collect(),
        _ => data.to_vec::<f32>().unwrap(),
    }
}
fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i|(((i*17+seed)%97) as f32-48.)/128.).collect()
}
fn runtime() {
    let name=std::any::type_name::<TestRuntime>();
    assert!(name.contains("CudaRuntime"),"real CUDA runtime required: {name}");
    assert_eq!(floats(tensor(vec![1.],[1],DType::F32)),vec![1.]);
    println!("RUDA_V36_HISTORY_COMPACTION_GPU_EXECUTED={name}");
}
struct Case {
    plan:DevicePlan<TestRuntime>, workspace:OrderedBackwardWorkspace<TestRuntime>,
    q:Tensor,k:Tensor,v:Tensor,qp:Tensor,kp:Tensor,g:Tensor,
    dq:Tensor,dk:Tensor,dv:Tensor,dqp:Tensor,dkp:Tensor,mla:bool,
}
impl Case {
    fn new(mla:bool,dtype:DType,mode:usize,n:usize,length:usize,d:usize,dv:usize,pos:usize,reserved:usize) -> Self {
        runtime();
        let page_size=16;let logical=length.div_ceil(page_size);let pages=(2*logical+reserved).max(1);
        let mut ids=Vec::new();let mut positions=Vec::new();
        for i in 0..n {
            let seq=i%3;ids.push(seq as u32);
            positions.push(if seq==2 || length==0 {0} else {match mode {
                1=>((i*37)%length) as u32,
                2=>(length-1) as u32,
                _=>(i/6).min(length-1) as u32,
            }});
        }
        let first=(0..logical as u32).collect::<Vec<_>>();
        let mut second=(logical as u32..2*logical as u32).collect::<Vec<_>>();
        if logical>=2 {second[1]=0;} // One physical page at TWO logical positions.
        let host=HostPlan::new(page_size,pages,&[first,second,vec![]],
            &[length as u32,length as u32,0],&ids,&positions).unwrap();
        let heads=4;let kh=if mla {1}else{2};let dv=if mla {d}else{dv};
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
    fn poison(&mut self) {
        let out=|like:&Tensor|tensor(vec![f32::NAN;like.meta.num_elements()],like.meta.shape().clone(),like.dtype);
        self.dq=out(&self.q);self.dk=out(&self.k);self.dv=out(&self.v);self.dqp=out(&self.qp);self.dkp=out(&self.kp);
    }
    fn run(&mut self,compact:bool,causal:bool,mask:u8)->Vec<Vec<f32>> {
        self.workspace.set_history_compaction(compact).unwrap();
        assert_eq!(self.workspace.history_compaction(),compact);
        let on=|bit:u8|mask&(1<<bit)!=0;
        if self.mla {
            unsafe{self.plan.mla_backward_ordered_into(&self.q,&self.qp,&self.k,&self.kp,&self.g,
                on(0).then_some(&self.dq),on(1).then_some(&self.dqp),on(2).then_some(&self.dk),on(3).then_some(&self.dkp),
                0.37,causal,&mut self.workspace)}.unwrap();
        } else {
            unsafe{self.plan.attention_backward_ordered_into(&self.q,&self.k,&self.v,&self.g,
                on(0).then_some(&self.dq),on(1).then_some(&self.dk),on(2).then_some(&self.dv),
                0.37,causal,&mut self.workspace)}.unwrap();
        }
        let list=if self.mla {vec![&self.dq,&self.dqp,&self.dk,&self.dkp]}else{vec![&self.dq,&self.dk,&self.dv]};
        list.into_iter().enumerate().filter(|(i,_)|on(*i as u8))
            .map(|(_,t)|floats(t.clone())).collect()
    }
}
fn assert_bits(a:&[Vec<f32>],b:&[Vec<f32>]) {
    assert_eq!(a.len(),b.len());
    for (a,b) in a.iter().zip(b) {
        assert!(a.iter().chain(b).all(|x|x.is_finite()),"invalid/poison history was loaded or output unwritten");
        assert_eq!(a.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),b.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
    }
}

fn compare(mla:bool,dtype:DType,mode:usize,n:usize,length:usize,reserved:usize,causal:bool,mask:u8,d:usize,dv:usize,p:usize) {
    let mut c=Case::new(mla,dtype,mode,n,length,d,dv,p,reserved);
    c.workspace.set_history_row_cache(false);
    c.poison();let expected=c.run(false,causal,mask);
    let before=c.workspace.bytes();let had_map=c.workspace.history_compaction_pages().is_some();
    c.poison();assert_bits(&expected,&c.run(true,causal,mask));
    let after=c.workspace.bytes();
    assert_eq!(after-before,if had_map {0}else{c.plan.host().pages*4});
    for (compact,cache,prune) in [(true,false,true),(true,true,true),(true,true,false),(false,false,true)] {
        c.workspace.set_history_row_cache(cache);c.workspace.set_query_pruning(prune);
        c.poison();assert_bits(&expected,&c.run(compact,causal,mask));
        assert_eq!(c.workspace.bytes(),after,"immutable page map must be retained and reused");
    }
}
#[test] fn v36_gqa_fp32_sparse(){compare(false,DType::F32,0,31,53,60,true,7,5,7,3);}
#[test] fn v36_gqa_fp16_sparse(){compare(false,DType::F16,2,31,53,60,true,7,128,128,3);}
#[test] fn v36_gqa_key_only(){compare(false,DType::F32,1,31,53,20,true,2,5,7,3);}
#[test] fn v36_gqa_value_only(){compare(false,DType::F32,1,31,53,20,true,4,5,129,3);}
#[test] fn v36_gqa_query_only(){compare(false,DType::F32,1,31,53,20,true,1,5,7,3);}
#[test] fn v36_gqa_noncausal(){compare(false,DType::F32,0,31,53,20,false,7,5,7,3);}
#[test] fn v36_gqa_empty_queries(){compare(false,DType::F32,0,0,53,20,true,7,5,7,3);}
#[test] fn v36_gqa_empty_histories(){compare(false,DType::F32,0,7,0,20,true,7,5,7,3);}
#[test] fn v36_gqa_full_occupancy(){compare(false,DType::F32,2,7,8,0,true,7,5,7,3);}
#[test] fn v36_gqa_request_without_queries(){compare(false,DType::F32,2,1,53,0,true,7,5,7,3);}
#[test] fn v36_gqa_wide_tail(){compare(false,DType::F32,2,5,17,9,true,7,129,257,3);}
#[test] fn v36_mla_fp32_sparse(){compare(true,DType::F32,0,31,53,20,true,15,5,5,3);}
#[test] fn v36_mla_fp16_sparse(){compare(true,DType::F16,2,31,53,20,true,15,128,128,64);}
#[test] fn v36_mla_latent_only(){compare(true,DType::F32,1,31,53,20,true,4,5,5,3);}
#[test] fn v36_mla_position_only(){compare(true,DType::F32,1,31,53,20,true,8,5,5,3);}
#[test] fn v36_mla_queries_only(){compare(true,DType::F32,1,31,53,20,true,3,5,5,3);}
#[test] fn v36_mla_noncausal(){compare(true,DType::F32,0,31,53,20,false,15,5,5,3);}
#[test] fn v36_mla_empty_histories(){compare(true,DType::F32,0,7,0,20,true,15,5,5,3);}
#[test] fn v36_mla_latent_512(){compare(true,DType::F32,2,5,17,9,true,15,512,512,64);}
#[test] fn v36_compaction_poison_inactive_inputs(){
    for mla in [false,true] {
        let mut c=Case::new(mla,DType::F32,2,17,33,5,5,3,20);
        let map=history_compaction::build(c.plan.host(),c.plan.host().pages).unwrap();
        let mut k=values(c.k.meta.num_elements(),3);let mut v=values(c.v.meta.num_elements(),5);let mut kp=values(c.kp.meta.num_elements(),11);
        for &page in &map.pages[map.active..] {
            let page=page as usize;let kstride=k.len()/map.pages.len();let vstride=v.len()/map.pages.len();let pstride=kp.len()/map.pages.len();
            k[page*kstride..(page+1)*kstride].fill(f32::NAN);
            v[page*vstride..(page+1)*vstride].fill(f32::NAN);
            kp[page*pstride..(page+1)*pstride].fill(f32::NAN);
        }
        c.k=tensor(k,c.k.meta.shape().clone(),c.k.dtype);
        c.v=if mla {c.k.clone()}else{tensor(v,c.v.meta.shape().clone(),c.v.dtype)};
        c.kp=tensor(kp,c.kp.meta.shape().clone(),c.kp.dtype);
        let mask=if mla {15}else{7};c.poison();let expected=c.run(false,true,mask);
        c.poison();assert_bits(&expected,&c.run(true,true,mask));
    }
}
#[test] fn v36_uniform_attention_matches_analytic_gradient(){
    let mut c=Case::new(false,DType::F32,2,1,2,3,5,3,6);
    c.q=tensor(vec![0.;c.q.meta.num_elements()],c.q.meta.shape().clone(),DType::F32);
    c.k=tensor(vec![0.;c.k.meta.num_elements()],c.k.meta.shape().clone(),DType::F32);
    c.v=tensor(vec![1.;c.v.meta.num_elements()],c.v.meta.shape().clone(),DType::F32);
    c.g=tensor(vec![1.;c.g.meta.num_elements()],c.g.meta.shape().clone(),DType::F32);
    let host=HostPlan::new(16,8,&[vec![6]],&[2],&[0],&[1]).unwrap();
    c.plan=DevicePlan::upload(host,&c.q);c.workspace=OrderedBackwardWorkspace::new(&c.plan,&c.q).unwrap();
    c.poison();let out=c.run(true,true,7);
    assert!(out[0].iter().chain(&out[1]).all(|&x|x==0.));
    for (i,&x) in out[2].iter().enumerate() {
        let row=i/5;let slot=row/2;let live=slot/16==6 && slot%16<2;
        // 4 query heads / 2 KV heads, two equally likely history tokens.
        assert_eq!(x,if live {1.}else{0.});
    }
}
#[test] fn bf16_history_compaction_v36(){
    compare(false,DType::BF16,2,11,17,16,true,7,128,128,3);
    compare(true,DType::BF16,2,11,17,16,true,15,128,128,64);
}
#[test]
#[ignore="explicit paired timing; includes backward, readback and synchronization"]
fn benchmark_history_compaction_v36(){
    use std::time::Instant;
    for (mla,n,length,reserved,d) in [(false,32usize,128usize,0usize,128usize),(false,32,128,240,128),(true,32,128,240,512)] {
        let mut c=Case::new(mla,DType::F16,2,n,length,d,d,64,reserved);let mask=if mla{15}else{7};
        // Allocate the map before ALL warmup/timing, for a symmetric comparison.
        c.workspace.set_history_compaction(true).unwrap();
        let expected=c.run(false,true,mask);assert_bits(&expected,&c.run(true,true,mask));
        for sample in 0..4 {for compact in if sample%2==0{[false,true]}else{[true,false]}{c.run(compact,true,mask);}}
        for sample in 0..7 {for compact in if sample%2==0{[false,true]}else{[true,false]}{
            let start=Instant::now();let actual=c.run(compact,true,mask);let seconds=start.elapsed().as_secs_f64();
            assert_bits(&expected,&actual);
            println!("RUDA_V36_HISTORY_COMPACTION_BENCH mla={mla} queries={n} length={length} reserved={reserved} dim={d} sample={sample} compact={compact} seconds_with_readback={seconds:.9}");
        }}
    }
}
