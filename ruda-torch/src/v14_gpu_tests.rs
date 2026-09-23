//! Real NVIDIA/PTX acceptance. Ignored by ordinary cargo test; explicitly run
//! with RUDA_REQUIRE_GPU=1 RUDA_CUDA_COMPILER=ptx cargo test -p
//! ruda-torch-native v14_gpu_ -- --ignored --test-threads=1
use super::*;
use ruda_core::tensor::{Shape,data::TensorData,DType};
use ruda_kernel::tensor::{RudaTensor,transfer::from_data,readback::into_data_sync};
use rudnn::paged_attention::{HostPlan,DevicePlan};
use rublas::tensor_grouped::{GroupedStrategy,grouped_matmul_nt,grouped_matmul_nt_segmented};
type Tensor=RudaTensor<CudaRuntime>;
fn required() {
    assert_eq!(std::env::var("RUDA_REQUIRE_GPU").as_deref(),Ok("1"));
    assert_eq!(std::env::var("RUDA_CUDA_COMPILER").as_deref(),Ok("ptx"));
}
fn floats(values:Vec<f32>,shape:impl Into<Shape>,half:bool)->Tensor {
    let shape=shape.into();
    let data=if half {TensorData::new(values.into_iter().map(f16::from_f32).collect::<Vec<_>>(),shape)}
        else {TensorData::new(values,shape)};
    from_data(data,&CudaDevice::default())
}
fn read(t:Tensor)->Vec<f32> {
    let dtype=t.dtype;let d=into_data_sync(t);
    if dtype==DType::F16 {d.to_vec::<f16>().unwrap().into_iter().map(f16::to_f32).collect()}
    else {d.to_vec::<f32>().unwrap()}
}
fn close(a:&[f32],b:&[f32],tol:f32) {
    assert_eq!(a.len(),b.len());
    for (&a,&b) in a.iter().zip(b) {assert!((a-b).abs()<=tol*(1.0+b.abs()),"{a} != {b}");}
}
#[test] #[ignore="requires an actual NVIDIA GPU and native PTX compiler"]
fn v14_gpu_paged_ragged_absolute_positions_and_append() {
    required();
    let q=floats(vec![0.0;3*4*33],[3,4,33],false);
    let k=floats(vec![0.0;4*4*2*33],[4,4,2,33],false);
    let values=(0..4*4*2*17).map(|i|(i/34) as f32).collect();
    let v=floats(values,[4,4,2,17],false);
    let host=HostPlan::new(4,4,&[vec![3,0],vec![2]],&[6,2],&[0,1,0],&[4,1,5]).unwrap();
    let plan=DevicePlan::upload(host,&q);
    let out=read(plan.attention(q,k,v,0.1,true).unwrap());
    let expected=[10.8,8.5,55.0/6.0].into_iter().flat_map(|v|vec![v;4*17]).collect::<Vec<_>>();
    close(&out,&expected,1e-5);
    // A fork remains untouched. Also validates the shared-handle COW guard.
    let host=HostPlan::new(4,1,&[vec![0]],&[1],&[0],&[0]).unwrap();
    let nk=floats(vec![3.0;2*33],[1,2,33],false);let nv=floats(vec![5.0;2*17],[1,2,17],false);
    let k=floats(vec![0.0;4*2*33],[1,4,2,33],false);let before=k.clone();
    let v=floats(vec![0.0;4*2*17],[1,4,2,17],false);
    let plan=DevicePlan::upload(host,&nk);let (updated,_)=plan.append(nk,nv,k,v).unwrap();
    assert!(read(before).iter().all(|&v|v==0.0));assert!(read(updated)[..66].iter().all(|&v|v==3.0));
}
#[test] #[ignore="requires an actual NVIDIA GPU and native PTX compiler"]
fn v14_gpu_mla_rank512() {
    required();let q=floats(vec![0.0;2*4*512],[2,4,512],true);
    let qp=floats(vec![0.0;2*4*64],[2,4,64],true);
    let c=floats((0..2*4*512).map(|i|(i/512) as f32).collect(),[2,4,1,512],true);
    let p=floats(vec![0.0;2*4*64],[2,4,1,64],true);
    let host=HostPlan::new(4,2,&[vec![1,0]],&[6],&[0,0],&[4,5]).unwrap();
    let plan=DevicePlan::upload(host,&q);let out=read(plan.mla(q,qp,c,p,192.0f32.sqrt().recip(),true).unwrap());
    let expected=[4.4,23.0/6.0].into_iter().flat_map(|x|vec![x;4*512]).collect::<Vec<_>>();
    close(&out,&expected,2e-3);
}
#[test] #[ignore="requires 16x16x16 FP16 cooperative matrix support; absence is a failure"]
fn v14_gpu_tensorcore_expert_tails() {
    required();let (m,n,k,e)=(19,33,35,4);
    let x=floats((0..m*k).map(|i|(i%17) as f32/17.0-0.5).collect(),[m,k],true);
    let w=floats((0..e*n*k).map(|i|(i%23) as f32/23.0-0.5).collect(),[e,n,k],true);
    let ids=from_data(TensorData::new((0..m).map(|i|if i<1{1u32}else{2u32}).collect::<Vec<_>>(),[m]),&CudaDevice::default());
    let offsets=from_data(TensorData::new(vec![0u32,0,1,m as u32,m as u32],[5]),&CudaDevice::default());
    let reference=read(grouped_matmul_nt(x.clone(),w.clone(),ids.clone()).unwrap());
    // SAFETY: these explicit device offsets/row IDs have been constructed to agree.
    let result=read(unsafe{grouped_matmul_nt_segmented(x,w,ids,offsets,GroupedStrategy::TensorCore)}.unwrap());
    close(&result,&reference,3e-3);
}
#[test] #[ignore="requires an actual NVIDIA GPU and native PTX compiler"]
fn v14_gpu_group_router_bias_selects_but_does_not_weight() {
    required();let logits=floats(vec![0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0],[1,8],false);
    let bias=floats(vec![10.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[8],false);
    let options=rudnn::moe::GroupRoutingOptions{top_k:2,groups:2,selected_groups:1,group_top_two:true,renormalize:true,scale:2.0};
    let plan=rudnn::moe::route_sigmoid_grouped(logits,Some(bias),options).unwrap();
    assert_eq!(into_data_sync(plan.expert_indices().clone()).to_vec::<u32>().unwrap(),vec![0,3]);
    let p=1.0/(1.0+(-3.0f32).exp());
    close(&read(plan.weights().clone()),&[1.0/(0.5+p),2.0*p/(0.5+p)],2e-6);
}
