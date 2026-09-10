//! Real GPU matmul integration checks, not mock timing tests. Explicitly opt in on matching hardware.
use ruda::runtime::{server::ComputeServer, tune::stack::*};
use ruda_tensor::{api::{Tensor,TensorData,backend::Backend}, DeviceOps};
use ruda_tensor_device::{DeviceBackend,DeviceRuntime};
use ruda_core::tensor::DType;
use std::sync::OnceLock;
type B<R> = DeviceBackend<R,f32,i32,u8>;
fn controller()->&'static StackTuner {
    static INIT:OnceLock<&'static StackTuner>=OnceLock::new();
    INIT.get_or_init(||enable_stack_autotune(StackPolicy {warmups:1,samples:3,max_candidates:8,
        tolerance:Tolerance {absolute:0.03,relative:0.03,..Tolerance::default()},..StackPolicy::default()},None).unwrap())
}
fn check<R>(dtype:DType)
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps+Default {
    let tuner=controller(); let device=R::Device::default();
    assert!(B::<R>::supports_dtype(&device,dtype),"hardware must support the requested dtype");
    for (m,n,k) in [(3,5,7),(17,9,11)] {
        let av:Vec<_>=(0..m*k).map(|i|((i%7)as f32-3.)*0.25).collect();
        let bv:Vec<_>=(0..k*n).map(|i|((i%5)as f32-2.)*0.25).collect();
        let mut expected=vec![0f64;m*n];
        for i in 0..m {for j in 0..n {for z in 0..k {expected[i*n+j]+=av[i*k+z]as f64*bv[z*n+j]as f64;}}}
        // Stored transposed, then exposed with non-contiguous strides for the real matmul.
        let transposed:Vec<_>=(0..k).flat_map(|z|(0..m).map({let av=&av;move |i|av[i*k+z]})).collect();
        let a=Tensor::<B<R>,2>::from_data(TensorData::new(transposed,[k,m]),&device).cast(dtype).swap_dims(0,1);
        let b=Tensor::<B<R>,2>::from_data(TensorData::new(bv,[k,n]),&device).cast(dtype);
        let out=a.clone().matmul(b.clone()).into_data().convert::<f32>().to_vec::<f32>().unwrap();
        ruda::runtime::tune::validate_finite_values(expected,out.iter().map(|&v|v as f64),0.03,0.03).unwrap();
        B::<R>::sync(&device).unwrap(); let before=tuner.stats();
        let second=a.matmul(b).into_data().convert::<f32>().to_vec::<f32>().unwrap();
        assert_eq!(out,second); assert!(tuner.stats().memory_hits>before.memory_hits);
        assert!(tuner.reports().iter().any(|report|report.operation.contains("rublas::matmul")));
    }
}
macro_rules! cases {
    ($name:ident,$runtime:ty)=> {mod $name {use super::*;
        #[test] #[ignore="requires real device and driver"] fn matmul_f32(){check::<$runtime>(DType::F32);}
        #[test] #[ignore="requires real device with F16 support"] fn matmul_f16(){check::<$runtime>(DType::F16);}
        #[test] #[ignore="requires real device with BF16 support"] fn matmul_bf16(){check::<$runtime>(DType::BF16);}
    }};
}
#[cfg(feature="nvidia")] cases!(nvidia,ruda_driver_cuda::CudaRuntime);
#[cfg(feature="amd")] cases!(amd,ruda_driver_hip::HipRuntime);
