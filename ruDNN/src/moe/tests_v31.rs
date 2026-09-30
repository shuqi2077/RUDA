//! Tests execute production GPU kernels; host arithmetic supplies the oracle.
use super::*;
use half::{bf16,f16};
use ruda_core::tensor::data::TensorData;
use ruda_kernel::{dsl::{calculate_ruda_count_elemwise,prelude::RudaDim},tensor::{transfer::from_data,readback::into_data_sync}};
use ruda_test_runtime::TestRuntime;
type Tensor=RudaTensor<TestRuntime>;
fn round(x:f32,d:DType)->f32{match d{DType::F16=>f16::from_f32(x).to_f32(),DType::BF16=>bf16::from_f32(x).to_f32(),_=>x}}
fn tensor(v:&[f32],shape:impl Into<Shape>,d:DType)->Tensor {
    let s=shape.into();let data=match d {
        DType::F16=>TensorData::new(v.iter().map(|&x|f16::from_f32(x)).collect::<Vec<_>>(),s),
        DType::BF16=>TensorData::new(v.iter().map(|&x|bf16::from_f32(x)).collect::<Vec<_>>(),s),
        _=>TensorData::new(v.to_vec(),s)};from_data(data,&Default::default())
}
fn floats(t:Tensor)->Vec<f32>{let d=t.dtype;let data=into_data_sync(t);match d {
    DType::F16=>data.to_vec::<f16>().unwrap().into_iter().map(f16::to_f32).collect(),
    DType::BF16=>data.to_vec::<bf16>().unwrap().into_iter().map(bf16::to_f32).collect(),
    _=>data.to_vec::<f32>().unwrap()}}
fn close(a:&[f32],b:&[f32],tol:f32){assert_eq!(a.len(),b.len());for(&x,&y)in a.iter().zip(b){assert!(x.is_finite()&&y.is_finite()&&(x-y).abs()<=tol*y.abs().max(1.),"{x}!={y}");}}
fn values(len:usize,seed:usize,d:DType)->Vec<f32>{(0..len).map(|i|round((((i*31+seed)%103)as f32-51.)/16.,d)).collect()}
fn activation(d:DType,len:usize){
    let gv=values(len,1,d);let uv=values(len,3,d);let yv=values(len,7,d);
    let gate=tensor(&gv,[1,len],d);let up=tensor(&uv,[1,len],d);let dy=tensor(&yv,[1,len],d);
    let out=empty(&gate,[1,len],d);let dg=empty(&gate,[1,len],d);let du=empty(&gate,[1,len],d);
    if len!=0 {let dim=RudaDim::new(gate.client.properties(),len);let count=calculate_ruda_count_elemwise(&gate.client,len,dim);
        kernels::experts::swiglu_out::launch::<TestRuntime>(&gate.client,count,dim,gate.clone().into_array_arg(),up.clone().into_array_arg(),out.clone().into_array_arg(),d.into());
        kernels::experts::swiglu_backward::launch::<TestRuntime>(&gate.client,calculate_ruda_count_elemwise(&gate.client,len,dim),dim,
            gate.clone().into_array_arg(),up.clone().into_array_arg(),dy.into_array_arg(),dg.clone().into_array_arg(),du.clone().into_array_arg(),d.into());
    }
    let mut expected=Vec::new();let mut eg=Vec::new();let mut eu=Vec::new();
    for i in 0..len{let g=gv[i];let s=1./(1.+(-g).exp());let activation=round(g/(1.+(-g).exp()),d);
        expected.push(round(activation*uv[i],d));eg.push(round(round(yv[i]*uv[i],d)*s*(1.+g*(1.-s)),d));eu.push(round(yv[i]*activation,d));}
    let tol=if d==DType::F32 {3e-5}else if d==DType::BF16{0.04}else{0.005};
    close(&floats(out),&expected,tol);close(&floats(dg),&eg,tol);close(&floats(du),&eu,tol);
    close(&floats(gate),&gv,0.);close(&floats(up),&uv,0.);
}
#[test]fn v31_runtime_marker(){let r=std::any::type_name::<TestRuntime>();assert!(r.contains("CudaRuntime"));activation(DType::F32,1);println!("RUDA_V31_EXPERT_GPU_EXECUTED={r}");}
#[test]fn v31_swiglu_fp32(){activation(DType::F32,257);}
#[test]fn v31_swiglu_fp16_rounding(){activation(DType::F16,4097);}
#[test]fn v31_swiglu_empty(){activation(DType::F16,0);}
#[test]fn bf16_expert_v31(){activation(DType::BF16,4097);}
#[test]fn v31_training_uses_selected_backward_strategy(){
    let d=DType::F16;let e=3;let hidden=5;let width=17;let tokens=19;
    let x=tensor(&values(tokens*hidden,13,d),[tokens,hidden],d);
    let logits=tensor(&(0..tokens*e).map(|i|if i%e==0{2.}else{-2.}).collect::<Vec<_>>(),[tokens,e],DType::F32);
    let packed=route(logits,RoutingOptions{top_k:1,renormalize:true}).unwrap().dispatch(x).unwrap();
    let experts=SwiGluExperts::new(tensor(&values(e*width*hidden,2,d),[e,width,hidden],d),
        tensor(&values(e*width*hidden,4,d),[e,width,hidden],d),tensor(&values(e*hidden*width,6,d),[e,hidden,width],d)).unwrap();
    let a=experts.forward_dispatched_training(&packed,GroupedStrategy::Scalar).unwrap();
    let b=experts.forward_dispatched_training(&packed,GroupedStrategy::Scalar).unwrap();
    close(&floats(a.output.clone()),&floats(b.output.clone()),0.);
    let g=tensor(&values(tokens*hidden,5,d),[tokens,hidden],d);
    let sa=a.cache.backward(g.clone()).unwrap();let tc=b.cache.backward_with_strategy(g,GroupedStrategy::TensorCore).unwrap();
    close(&floats(tc.dinput),&floats(sa.dinput),0.03);
    for (actual,expected) in [(tc.dgate,sa.dgate),(tc.dup,sa.dup),(tc.ddown,sa.ddown)] {
        assert_eq!(actual.dtype,DType::F32);close(&floats(actual),&floats(expected),0.03);
    }
}
#[test]fn v31_forward_new_output_matches_legacy_inplace(){
    let dtype=DType::F16;let len=257;
    let g=tensor(&values(len,23,dtype),[1,len],dtype);let u=tensor(&values(len,9,dtype),[1,len],dtype);
    let old=g.copy();let out=empty(&g,[1,len],dtype);let dim=RudaDim::new(g.client.properties(),len);
    kernels::experts::swiglu::launch::<TestRuntime>(&g.client,calculate_ruda_count_elemwise(&g.client,len,dim),dim,
        old.clone().into_array_arg(),u.clone().into_array_arg(),dtype.into());
    kernels::experts::swiglu_out::launch::<TestRuntime>(&g.client,calculate_ruda_count_elemwise(&g.client,len,dim),dim,
        g.clone().into_array_arg(),u.into_array_arg(),out.clone().into_array_arg(),dtype.into());
    close(&floats(out),&floats(old),0.);
}
