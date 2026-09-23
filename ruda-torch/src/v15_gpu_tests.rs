//! Required real-device acceptance for v15. Host reference tests do not run it.
use super::*;
use ruda_core::tensor::{Shape, data::TensorData, DType};
use ruda_kernel::tensor::{RudaTensor, transfer::from_data, readback::into_data_sync};
use rudnn::paged_attention::{HostPlan, DevicePlan, SplitWorkspace};
type Tensor=RudaTensor<CudaRuntime>;
fn required() {
    assert_eq!(std::env::var("RUDA_REQUIRE_GPU").as_deref(),Ok("1"));
    assert_eq!(std::env::var("RUDA_CUDA_COMPILER").as_deref(),Ok("ptx"));
}
fn floats(v:Vec<f32>,shape:impl Into<Shape>,dtype:DType)->Tensor {
    let shape=shape.into();
    let data=match dtype {
        DType::F16=>TensorData::new(v.into_iter().map(f16::from_f32).collect::<Vec<_>>(),shape),
        DType::BF16=>TensorData::new(v.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),shape),
        _=>TensorData::new(v,shape),
    };
    from_data(data,&CudaDevice::default())
}
fn read(t:Tensor)->Vec<f32> {
    let dtype=t.dtype;let data=into_data_sync(t);
    match dtype {
        DType::F16=>data.to_vec::<f16>().unwrap().into_iter().map(f16::to_f32).collect(),
        DType::BF16=>data.to_vec::<bf16>().unwrap().into_iter().map(bf16::to_f32).collect(),
        _=>data.to_vec::<f32>().unwrap(),
    }
}
fn close(a:&[f32],b:&[f32],tol:f32) {
    assert_eq!(a.len(),b.len());
    for (&x,&y) in a.iter().zip(b) { assert!(x.is_finite() && y.is_finite() && (x-y).abs()<=tol*(1.0+y.abs()),"{x} != {y}"); }
}
fn wave(n:usize,seed:usize)->Vec<f32> { (0..n).map(|i| ((i*13+seed)%41) as f32/41.0-0.5).collect() }

#[test] #[ignore="requires real RUDA/PTX GPU execution"]
fn v15_gpu_split_ragged_dtype_tails() {
    required();
    for dtype in [DType::F32,DType::F16,DType::BF16] {
        let q=floats(wave(4*4*33,1),[4,4,33],dtype);
        let k=floats(wave(7*7*2*33,2),[7,7,2,33],dtype);
        let v=floats(wave(7*7*2*17,3),[7,7,2,17],dtype);
        let plan=DevicePlan::upload(HostPlan::new(7,7,&[vec![4,1,6,0],vec![5,3],vec![]],
            &[25,9,0],&[0,1,0,2],&[13,8,24,0]).unwrap(),&q);
        for causal in [false,true] {
            let reference=read(plan.attention(q.clone(),k.clone(),v.clone(),33.0f32.sqrt().recip(),causal).unwrap());
            for splits in [2,3,8,32] {
                let out=floats(vec![0.0;4*4*17],[4,4,17],dtype);
                let mut work=SplitWorkspace::new(&q,4,4,17,splits).unwrap();
                // SAFETY: fresh output and private same-queue workspace.
                unsafe{plan.attention_into_split(&q,&k,&v,None,&out,33.0f32.sqrt().recip(),causal,&mut work)}.unwrap();
                let actual=read(out);close(&actual,&reference,if dtype==DType::F32{5e-5}else{1e-2});
                assert!(actual[3*4*17..].iter().all(|&x|x==0.0));
            }
        }
    }
}

#[test] #[ignore="requires real RUDA/PTX GPU execution"]
fn v15_gpu_split_mla_rank512() {
    required();let dtype=DType::F16;
    let q=floats(wave(2*4*512,1),[2,4,512],dtype);
    let qp=floats(wave(2*4*64,2),[2,4,64],dtype);
    let c=floats(wave(3*7*512,3),[3,7,1,512],dtype);
    let kp=floats(wave(3*7*64,4),[3,7,1,64],dtype);
    let host=HostPlan::new(7,3,&[vec![2,0,1]],&[17],&[0,0],&[13,16]).unwrap();
    let plan=DevicePlan::upload(host,&q);let scale=192.0f32.sqrt().recip();
    let expected=read(plan.mla(q.clone(),qp.clone(),c.clone(),kp.clone(),scale,true).unwrap());
    let mut work=SplitWorkspace::new(&q,2,4,512,8).unwrap();
    for _ in 0..3 {
        let out=floats(vec![0.0;2*4*512],[2,4,512],dtype);
        unsafe{plan.attention_into_split(&q,&c,&c,Some((&qp,&kp)),&out,scale,true,&mut work)}.unwrap();
        close(&read(out),&expected,5e-3);
    }
}

#[test] #[ignore="requires real RUDA/PTX GPU execution"]
fn v15_gpu_split_workspace_reuse_with_shorter_schedule() {
    required();let dtype=DType::F32;
    let q=floats(vec![0.0;33],[1,1,33],dtype);let k=floats(vec![0.0;3*7*33],[3,7,1,33],dtype);
    let v=floats((0..21).map(|x|x as f32).collect(),[3,7,1,1],dtype);
    let long=DevicePlan::upload(HostPlan::new(7,3,&[vec![2,1,0]],&[21],&[0],&[20]).unwrap(),&q);
    let short=DevicePlan::upload(HostPlan::new(7,3,&[vec![2]],&[1],&[0],&[0]).unwrap(),&q);
    let empty=DevicePlan::upload(HostPlan::new(7,3,&[vec![]],&[0],&[0],&[0]).unwrap(),&q);
    let mut work=SplitWorkspace::new(&q,1,1,1,32).unwrap();
    for (plan,expected) in [(&long,10.0),(&short,14.0),(&empty,0.0)] {
        let out=floats(vec![-1.0],[1,1,1],dtype);
        unsafe{plan.attention_into_split(&q,&k,&v,None,&out,1.0,true,&mut work)}.unwrap();
        close(&read(out),&[expected],1e-5);
    }
}

#[test] #[ignore="requires actual GPU memory-manager handle counts"]
fn v15_gpu_append_unique_in_place() {
    required();
    let k=floats(vec![0.0;16],[1,4,1,4],DType::F32);
    let v=floats(vec![0.0;16],[1,4,1,4],DType::F32);
    let nk=floats(vec![3.0;4],[1,1,4],DType::F32);let nv=floats(vec![5.0;4],[1,1,4],DType::F32);
    let plan=DevicePlan::upload(HostPlan::new(4,1,&[vec![0]],&[1],&[0],&[0]).unwrap(),&nk);
    assert!(k.can_mut() && v.can_mut(),"unexpected externally shared initial allocation");
    let (k,v,report)=plan.append_with_report(nk,nv,k,v).unwrap();
    assert!(!report.key_copied && !report.value_copied);assert_eq!(report.copied_bytes,0);
    assert_eq!(&read(k)[..4],&[3.0;4]);assert_eq!(&read(v)[..4],&[5.0;4]);
}

#[test] #[ignore="requires actual GPU memory-manager handle counts"]
fn v15_gpu_append_preserves_fork() {
    required();let k=floats(vec![0.0;16],[1,4,1,4],DType::F32);let snapshot=k.clone();
    let v=floats(vec![0.0;16],[1,4,1,4],DType::F32);
    let nk=floats(vec![3.0;4],[1,1,4],DType::F32);let nv=floats(vec![5.0;4],[1,1,4],DType::F32);
    let plan=DevicePlan::upload(HostPlan::new(4,1,&[vec![0]],&[1],&[0],&[0]).unwrap(),&nk);
    assert!(!k.can_mut() && v.can_mut());
    let (k,v,report)=plan.append_with_report(nk,nv,k,v).unwrap();
    assert!(report.key_copied && !report.value_copied);assert_eq!(report.copied_bytes,64);
    assert!(read(snapshot).iter().all(|&x|x==0.0));assert_eq!(&read(k)[..4],&[3.0;4]);assert_eq!(&read(v)[..4],&[5.0;4]);
}

#[test] #[ignore="requires actual GPU memory-manager handle counts"]
fn v15_gpu_append_k_v_alias_and_empty_noop() {
    required();let k=floats(vec![0.0;16],[1,4,1,4],DType::F32);let v=k.clone();
    let nk=floats(vec![3.0;4],[1,1,4],DType::F32);let nv=floats(vec![5.0;4],[1,1,4],DType::F32);
    let plan=DevicePlan::upload(HostPlan::new(4,1,&[vec![0]],&[1],&[0],&[0]).unwrap(),&nk);
    let (k,v,report)=plan.append_with_report(nk,nv,k,v).unwrap();assert!(report.key_copied && report.value_copied);
    assert_eq!(&read(k)[..4],&[3.0;4]);assert_eq!(&read(v)[..4],&[5.0;4]);
    let k=floats(vec![2.0;16],[1,4,1,4],DType::F32);let snapshot=k.clone();let v=k.clone();
    let nk=floats(vec![],[0,1,4],DType::F32);let nv=floats(vec![],[0,1,4],DType::F32);
    let plan=DevicePlan::upload(HostPlan::new(4,1,&[vec![0]],&[1],&[],&[]).unwrap(),&nk);
    let (_,_,report)=plan.append_with_report(nk,nv,k,v).unwrap();assert_eq!(report.copied_bytes,0);
    assert!(read(snapshot).iter().all(|&x|x==2.0));
}
