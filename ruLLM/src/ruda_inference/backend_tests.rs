//! These tests execute real backend kernels; they are opt-in and need hardware.
use super::*;
use ruda_tensor::api::TensorData;
use crate::runtime::{NumericalTolerance, stable_rms_norm};

type B<R> = DeviceBackend<R, f32, i32, u8>;
fn host<R, const D: usize>(tensor: Tensor<B<R>, D>) -> Vec<f32>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
{
    tensor.into_data().convert::<f32>().to_vec::<f32>().unwrap()
}
fn tolerance(dtype: DType) -> NumericalTolerance {
    match dtype {
        DType::BF16 => NumericalTolerance { absolute: 0.025, relative: 0.025 },
        DType::F16 => NumericalTolerance { absolute: 0.004, relative: 0.004 },
        _ => NumericalTolerance { absolute: 1e-5, relative: 1e-4 },
    }
}
fn check_norms<R>(dtype: DType)
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps + Default,
{
    let device = R::Device::default();
    assert!(specialized_dtype::<R>(&device,dtype), "test device must support requested arithmetic dtype");
    for width in [1,7,31,32,33,63,64,65,127,128,129,896] {
        for (batch,sequence) in [(1,1),(2,7)] {
            let values: Vec<_> = (0..batch*sequence*width).map(|i| ((i%197) as f32-98.0)/31.0).collect();
            let gamma: Vec<_> = (0..width).map(|i| 0.5+(i%19) as f32/23.0).collect();
            let input = Tensor::<B<R>,3>::from_data(TensorData::new(values,[batch,sequence,width]),&device).cast(dtype).swap_dims(0,1);
            let gamma = Tensor::<B<R>,1>::from_data(TensorData::new(gamma,[width]),&device).cast(dtype);
            let input_values = host(input.clone()); let gamma_values = host(gamma.clone());
            let gamma64:Vec<_> = gamma_values.iter().map(|&v|v as f64).collect();
            let expected:Vec<_> = input_values.chunks_exact(width).flat_map(|row| {
                stable_rms_norm(&row.iter().map(|&v|v as f64).collect::<Vec<_>>(),&gamma64,1e-6).unwrap()
            }).collect();
            let actual:Vec<_> = host(rms_norm(input.clone(),gamma.clone(),1e-6)).iter().map(|&v|v as f64).collect();
            tolerance(dtype).compare(&actual,&expected).unwrap();
            let zero=Tensor::zeros_like(&input);
            let (_,normalized)=residual_rms_norm(input,zero,gamma,1e-6);
            let fused:Vec<_>=host(normalized).iter().map(|&v|v as f64).collect();
            tolerance(dtype).compare(&fused,&expected).unwrap();
        }
    }
}
fn check_overflow_norm<R>()
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps + Default,
{
    let device=R::Device::default();
    let values=vec![1e30f32;129];
    let x=Tensor::<B<R>,3>::from_data(TensorData::new(values,[1,1,129]),&device);
    let g=Tensor::<B<R>,1>::ones([129],&device);
    let actual=host(rms_norm(x,g,1e-6));
    for value in actual {assert!(value.is_finite()&&(value-1.0).abs()<1e-5);}
}
fn round(value: f32,dtype:DType)->f32 {
    match dtype { DType::BF16=>half::bf16::from_f32(value).to_f32(),DType::F16=>half::f16::from_f32(value).to_f32(),_=>value }
}
fn check_decode<R>(dtype:DType)
where R: DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps+Default,
{
    let device=R::Device::default();
    let width=reduction_width::<R>(&device,dtype,8).expect("test needs fixed subgroup") as usize;
    let dimension=2*width;
    let (batch,qheads,kvheads,capacity,length)=(2,4,2,17,7);
    let q:Vec<_>=(0..batch*qheads*dimension).map(|i|round(((i%29)as f32-14.0)/47.0,dtype)).collect();
    let k:Vec<_>=(0..batch*kvheads*capacity*dimension).map(|i|round(((i%23)as f32-11.0)/37.0,dtype)).collect();
    let v:Vec<_>=(0..batch*kvheads*capacity*dimension).map(|i|round(((i%31)as f32-15.0)/19.0,dtype)).collect();
    let query=Tensor::<B<R>,4>::from_data(TensorData::new(q.clone(),[batch,qheads,1,dimension]),&device).cast(dtype);
    let key=Tensor::<B<R>,4>::from_data(TensorData::new(k.clone(),[batch,kvheads,capacity,dimension]),&device).cast(dtype);
    let value=Tensor::<B<R>,4>::from_data(TensorData::new(v.clone(),[batch,kvheads,capacity,dimension]),&device).cast(dtype);
    let actual=host(gqa_decode_attention(query,key,value,length));
    let mut expected=Vec::new();
    for b in 0..batch {for head in 0..qheads {
        let qstart=(b*qheads+head)*dimension;
        let start=(b*kvheads+head/(qheads/kvheads))*capacity*dimension;
        let scores:Vec<_>=(0..length).map(|token| {
            let mut dot=0.0f32;
            for d in 0..dimension {dot+=q[qstart+d]*k[start+token*dimension+d];}
            round(round(dot,dtype)/(dimension as f32).sqrt(),dtype)
        }).collect();
        let maximum=scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);
        let denominator:f32=scores.iter().map(|s|(s-maximum).exp()).sum();
        let p:Vec<_>=scores.iter().map(|s|round((s-maximum).exp()/denominator,dtype)).collect();
        for d in 0..dimension {
            let mut sum=0.0f32;
            for token in 0..length {sum+=p[token]*v[start+token*dimension+d];}
            expected.push(round(sum,dtype)as f64);
        }
    }}
    tolerance(dtype).compare(&actual.iter().map(|&v|v as f64).collect::<Vec<_>>(),&expected).unwrap();
}
macro_rules! runtime_matrix {
    ($name:ident,$runtime:ty) => {
        mod $name {
            use super::*;
            #[test] #[ignore="requires real GPU, driver and toolchain"]
            fn norm_f32_shapes_and_strides(){check_norms::<$runtime>(DType::F32);}
            #[test] #[ignore="requires real GPU with F16 arithmetic"]
            fn norm_f16_shapes_and_strides(){check_norms::<$runtime>(DType::F16);}
            #[test] #[ignore="requires real GPU with BF16 arithmetic"]
            fn norm_bf16_shapes_and_strides(){check_norms::<$runtime>(DType::BF16);}
            #[test] #[ignore="requires real GPU, driver and toolchain"]
            fn rms_overflow(){check_overflow_norm::<$runtime>();}
            #[test] #[ignore="requires real GPU, driver and toolchain"]
            fn decode_f32_device_subgroup(){check_decode::<$runtime>(DType::F32);}
            #[test] #[ignore="requires real GPU with BF16 arithmetic"]
            fn decode_bf16_device_subgroup(){check_decode::<$runtime>(DType::BF16);}
        }
    }
}
#[cfg(feature="nvidia")]
runtime_matrix!(nvidia,ruda_driver_cuda::CudaRuntime);
#[cfg(feature="amd")]
runtime_matrix!(amd,ruda_driver_hip::HipRuntime);
