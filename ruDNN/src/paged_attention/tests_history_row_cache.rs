//! Real-device cached/uncached comparisons. No host or alternate runtime.
//! Inputs, outputs and workspace are resident and reused across both modes.
use super::*;
use half::{bf16, f16};
use ruda_core::tensor::data::TensorData;
use ruda_kernel::tensor::{transfer::from_data, readback::into_data_sync};
use ruda_test_runtime::TestRuntime;
type Tensor = RudaTensor<TestRuntime>;

fn tensor(values: Vec<f32>, shape: impl Into<Shape>, dtype: DType) -> Tensor {
    let shape=shape.into();
    let data=match dtype {
        DType::F16 => TensorData::new(values.into_iter().map(f16::from_f32).collect::<Vec<_>>(),shape),
        DType::BF16 => TensorData::new(values.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),shape),
        _ => TensorData::new(values,shape),
    };
    from_data(data,&Default::default())
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
    println!("RUDA_V35_HISTORY_CACHE_GPU_EXECUTED={name}");
}
struct Case {
    plan:DevicePlan<TestRuntime>, workspace:OrderedBackwardWorkspace<TestRuntime>,
    q:Tensor,k:Tensor,v:Tensor,qp:Tensor,kp:Tensor,g:Tensor,
    dq:Tensor,dk:Tensor,dv:Tensor,dqp:Tensor,dkp:Tensor,mla:bool,
}
impl Case {
    fn new(mla:bool,dtype:DType,mode:usize,n:usize,length:usize,d:usize,dv:usize,pos:usize) -> Self {
        runtime();
        let page_size=16;let logical=length.div_ceil(page_size);let pages=2*logical+2;
        let mut ids=Vec::new();let mut positions=Vec::new();
        for i in 0..n {
            let seq=i%3;ids.push(seq as u32);
            positions.push(if seq==2 {0} else {match mode {
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
    fn run(&mut self,cached:bool,causal:bool,mask:u8)->Vec<Vec<f32>> {
        self.workspace.set_history_row_cache(cached);
        assert_eq!(self.workspace.history_row_cache(),cached);
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
fn compare(mla:bool,dtype:DType,mode:usize,n:usize,causal:bool,mask:u8,d:usize,dv:usize,p:usize) {
    let mut c=Case::new(mla,dtype,mode,n,53,d,dv,p);
    let bytes=c.workspace.bytes();
    let expected=c.run(false,causal,mask);
    assert_bits(&expected,&c.run(true,causal,mask));
    assert_eq!(bytes,c.workspace.bytes(),"row cache must not grow global workspace");
    assert_bits(&expected,&c.run(true,causal,mask));
    assert_bits(&expected,&c.run(false,causal,mask));
}
#[test] fn v35_gqa_fp32_all(){compare(false,DType::F32,0,65,true,7,5,7,3);}
#[test] fn v35_gqa_fp16_all(){compare(false,DType::F16,0,65,true,7,128,128,3);}
#[test] fn v35_gqa_dk_only(){compare(false,DType::F32,1,65,true,2,5,7,3);}
#[test] fn v35_gqa_dv_only(){compare(false,DType::F32,1,65,true,4,5,7,3);}
#[test] fn v35_gqa_dq_only(){compare(false,DType::F32,1,65,true,1,5,7,3);}
#[test] fn v35_gqa_noncausal(){compare(false,DType::F32,1,65,false,7,5,7,3);}
#[test] fn v35_gqa_unsorted(){compare(false,DType::F32,1,65,true,7,5,7,3);}
#[test] fn v35_gqa_empty(){compare(false,DType::F32,0,0,true,7,5,7,3);}
#[test] fn v35_gqa_wide_tail(){compare(false,DType::F32,2,11,true,7,129,257,3);}
#[test] fn v35_mla_fp32_all(){compare(true,DType::F32,0,65,true,15,5,5,3);}
#[test] fn v35_mla_fp16_all(){compare(true,DType::F16,0,65,true,15,128,128,64);}
#[test] fn v35_mla_latent_only(){compare(true,DType::F32,1,65,true,4,5,5,3);}
#[test] fn v35_mla_position_only(){compare(true,DType::F32,1,65,true,8,5,5,3);}
#[test] fn v35_mla_queries_only(){compare(true,DType::F32,1,65,true,3,5,5,3);}
#[test] fn v35_mla_noncausal(){compare(true,DType::F32,1,65,false,15,5,5,3);}
#[test] fn v35_mla_empty(){compare(true,DType::F32,0,0,true,15,5,5,3);}
#[test] fn v35_mla_latent_512(){compare(true,DType::F32,2,11,true,15,512,512,64);}
#[test] fn v35_cache_does_not_survive_a_launch(){
    let mut c=Case::new(false,DType::F32,2,17,33,5,7,3);
    let old=c.run(true,true,7);
    c.k=tensor(values(c.k.meta.num_elements(),41),c.k.meta.shape().clone(),c.k.dtype);
    c.v=tensor(values(c.v.meta.num_elements(),59),c.v.meta.shape().clone(),c.v.dtype);
    let fresh=c.run(false,true,7);assert_bits(&fresh,&c.run(true,true,7));
    assert_ne!(old,fresh);
}
#[test] fn v35_cache_poison_reserved_pages(){
    let mut c=Case::new(false,DType::F32,2,17,33,5,7,3);
    let mut k=values(c.k.meta.num_elements(),3);let mut v=values(c.v.meta.num_elements(),5);
    let ke=k.len()-2*16*2*5;let ve=v.len()-2*16*2*7;
    k[ke..].fill(f32::NAN);v[ve..].fill(f32::NAN);
    c.k=tensor(k,c.k.meta.shape().clone(),c.k.dtype);
    c.v=tensor(v,c.v.meta.shape().clone(),c.v.dtype);
    let a=c.run(false,true,7);let b=c.run(true,true,7);assert_bits(&a,&b);
    assert!(b[1][ke..].iter().all(|&x|x==0.));assert!(b[2][ve..].iter().all(|&x|x==0.));
}
#[test] fn bf16_history_cache_v35(){
    compare(false,DType::BF16,1,65,true,7,128,128,3);
    compare(true,DType::BF16,1,65,true,15,128,128,64);
}
#[test]
#[ignore="explicit paired timing after correctness; includes readback and synchronization"]
fn benchmark_history_cache_v35(){
    use std::time::Instant;
    for (mla,n,length,d) in [(false,32usize,128usize,128usize),(false,128,128,128),(true,64,128,512)] {
        let mut c=Case::new(mla,DType::F16,2,n,length,d,d,64);
        let mask=if mla{15}else{7};
        let expected=c.run(false,true,mask);assert_bits(&expected,&c.run(true,true,mask));
        for sample in 0..4 { for cached in if sample%2==0{[false,true]}else{[true,false]} { c.run(cached,true,mask); } }
        for sample in 0..7 { for cached in if sample%2==0{[false,true]}else{[true,false]} {
            let start=Instant::now();let actual=c.run(cached,true,mask);let seconds=start.elapsed().as_secs_f64();
            assert_bits(&expected,&actual);
            println!("RUDA_V35_HISTORY_CACHE_BENCH mla={mla} queries={n} length={length} dim={d} sample={sample} cache={cached} seconds_with_readback={seconds:.9}");
        }}
    }
}
