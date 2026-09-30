//! Real device tests. CPU references are only the oracle, never the device path.
#![allow(unsafe_code)]
use super::*;
use half::{bf16,f16};
use ruda_core::tensor::data::TensorData;
use ruda_kernel::tensor::{transfer::from_data,readback::into_data_sync,permutation::swap_dims};
use ruda_test_runtime::TestRuntime;

type Tensor = RudaTensor<TestRuntime>;
fn round(v:f32,d:DType)->f32 { match d { DType::F16=>f16::from_f32(v).to_f32(),
    DType::BF16=>bf16::from_f32(v).to_f32(), _=>v } }
fn tensor(v:&[f32],shape:impl Into<Shape>,d:DType)->Tensor {
    let shape=shape.into();
    let data=match d { DType::F16=>TensorData::new(v.iter().map(|&x|f16::from_f32(x)).collect::<Vec<_>>(),shape),
        DType::BF16=>TensorData::new(v.iter().map(|&x|bf16::from_f32(x)).collect::<Vec<_>>(),shape),
        _=>TensorData::new(v.to_vec(),shape) };
    from_data(data,&Default::default())
}
fn floats(t:Tensor)->Vec<f32> { let d=t.dtype;let data=into_data_sync(t);match d {
    DType::F16=>data.to_vec::<f16>().unwrap().into_iter().map(f16::to_f32).collect(),
    DType::BF16=>data.to_vec::<bf16>().unwrap().into_iter().map(bf16::to_f32).collect(),
    _=>data.to_vec::<f32>().unwrap() } }
fn close(a:&[f32],b:&[f32],tol:f32) { assert_eq!(a.len(),b.len());
    for (i,(&x,&y)) in a.iter().zip(b).enumerate() {
        assert!(x.is_finite() && y.is_finite() && (x-y).abs()<=tol*y.abs().max(1.),"{i}: {x} != {y}");
    }
}
fn data(n:usize,seed:usize,d:DType)->Vec<f32> {
    (0..n).map(|i| round((((i*17+seed)%67) as f32-33.)/64.,d)).collect()
}
fn case(lengths:&[usize],k:usize,n:usize,d:DType,s:GroupedStrategy,noncontiguous:bool)->(Vec<f32>,Vec<f32>) {
    let e=lengths.len();let m=lengths.iter().sum::<usize>();
    let x=data(m*k,3,d);let w=data(e*n*k,5,d);let g=data(m*n,11,d);
    let mut prefix=vec![0u32];let mut ids=Vec::new();
    for (expert,&len) in lengths.iter().enumerate() {
        ids.extend(std::iter::repeat_n(expert as u32,len));prefix.push(prefix.last().unwrap()+len as u32);
    }
    let (xt,wt,gt)=if noncontiguous {
        let mut xx=vec![0.;m*k];let mut ww=vec![0.;e*n*k];let mut gg=vec![0.;m*n];
        for i in 0..m { for j in 0..k {xx[j*m+i]=x[i*k+j];} for j in 0..n {gg[j*m+i]=g[i*n+j];} }
        for a in 0..e {for j in 0..n {for b in 0..k {ww[(a*k+b)*n+j]=w[(a*n+j)*k+b];}}}
        (swap_dims(tensor(&xx,[k,m],d),0,1),swap_dims(tensor(&ww,[e,k,n],d),1,2),swap_dims(tensor(&gg,[n,m],d),0,1))
    } else {(tensor(&x,[m,k],d),tensor(&w,[e,n,k],d),tensor(&g,[m,n],d))};
    let result=unsafe {grouped_matmul_nt_backward_segmented_with_strategy(xt,wt,gt,
        from_data(TensorData::new(ids.clone(),[m]),&Default::default()),
        from_data(TensorData::new(prefix.clone(),[e+1]),&Default::default()),s)}.unwrap();
    assert_eq!(result.dinput.dtype,d);assert_eq!(result.dweights.dtype,DType::F32);
    let mut dx=vec![0.;m*k];let mut dw=vec![0.;e*n*k];
    for row in 0..m {let expert=ids[row] as usize;for col in 0..k {
        dx[row*k+col]=round((0..n).map(|j|g[row*n+j] as f64*w[(expert*n+j)*k+col] as f64).sum::<f64>() as f32,d);
    }}
    for a in 0..e {for j in 0..n {for b in 0..k {
        dw[(a*n+j)*k+b]=(prefix[a] as usize..prefix[a+1] as usize)
            .map(|r|g[r*n+j] as f64*x[r*k+b] as f64).sum::<f64>() as f32;
    }}}
    let actual_x=floats(result.dinput);let actual_w=floats(result.dweights);
    let tol=if d==DType::BF16 {0.04} else if d==DType::F16 {0.006} else {2e-5};
    close(&actual_x,&dx,tol);close(&actual_w,&dw,if s==GroupedStrategy::Scalar {2e-5} else {0.006});
    (actual_x,actual_w)
}
#[test] fn v31_runtime_marker() {
    let name=std::any::type_name::<TestRuntime>();assert!(name.contains("CudaRuntime"),"must use real CUDA runtime: {name}");
    close(&floats(tensor(&[1.],[1,1],DType::F32)),&[1.],0.);
    println!("RUDA_V31_GROUPED_GPU_EXECUTED={name}");
}
#[test] fn v31_scalar_reference_fp32(){case(&[1,0,17],9,19,DType::F32,GroupedStrategy::Scalar,false);}
#[test] fn v31_cooperative_one_element(){case(&[1],1,1,DType::F16,GroupedStrategy::TensorCore,false);}
#[test] fn v31_cooperative_aligned(){case(&[16,32],32,16,DType::F16,GroupedStrategy::TensorCore,false);}
#[test] fn v31_cooperative_all_tails(){case(&[1,3,17],17,33,DType::F16,GroupedStrategy::TensorCore,false);}
#[test] fn v31_cooperative_empty_experts(){case(&[0,0,17,0],19,35,DType::F16,GroupedStrategy::TensorCore,false);}
#[test] fn v31_cooperative_all_empty(){let (x,w)=case(&[0,0,0],17,19,DType::F16,GroupedStrategy::TensorCore,false);assert!(x.is_empty());assert!(w.iter().all(|&v|v==0.));}
#[test] fn v31_cooperative_unbalanced(){case(&[1,65,2,0],35,7,DType::F16,GroupedStrategy::TensorCore,false);}
#[test] fn v31_cooperative_noncontiguous(){case(&[3,0,19],17,33,DType::F16,GroupedStrategy::TensorCore,true);}
#[test] fn v31_auto_fp32_uses_scalar(){let a=case(&[17,0,3],19,11,DType::F32,GroupedStrategy::Scalar,false);let b=case(&[17,0,3],19,11,DType::F32,GroupedStrategy::Auto,false);assert_eq!(a,b);}
#[test] fn v31_auto_fp16_matches_forced(){let a=case(&[17,0,3],19,11,DType::F16,GroupedStrategy::TensorCore,false);let b=case(&[17,0,3],19,11,DType::F16,GroupedStrategy::Auto,false);assert_eq!(a,b);}
#[test] fn v31_cooperative_repeatable(){let a=case(&[19,7,0],33,17,DType::F16,GroupedStrategy::TensorCore,false);let b=case(&[19,7,0],33,17,DType::F16,GroupedStrategy::TensorCore,false);assert_eq!(a,b);}
#[test] fn v31_forced_fp32_rejected(){
    let result=unsafe {grouped_matmul_nt_backward_segmented_with_strategy(tensor(&[1.],[1,1],DType::F32),
        tensor(&[1.],[1,1,1],DType::F32),tensor(&[1.],[1,1],DType::F32),
        from_data(TensorData::new(vec![0u32],[1]),&Default::default()),
        from_data(TensorData::new(vec![0u32,1],[2]),&Default::default()),GroupedStrategy::TensorCore)};
    assert!(result.unwrap_err().0.contains("unavailable"));
}
// Optional BF16 acceptance is a separate test selection, never a silent skip.
#[test] fn bf16_grouped_v31(){case(&[1,0,17],19,33,DType::BF16,GroupedStrategy::TensorCore,false);}

#[test]
#[ignore = "explicit performance run after correctness; not GPU acceptance"]
fn benchmark_grouped_backward_v31() {
    use std::time::Instant;
    // Identical inputs/allocations and readback boundaries for both strategies.
    // End-to-end API time includes output allocation and transfer, not kernel-only time.
    for (e,rows,k,n) in [(8usize,4usize,128usize,128usize),(8,17,256,256),(4,65,128,256)] {
        let m=e*rows;let dtype=DType::F16;
        let x=tensor(&data(m*k,3,dtype),[m,k],dtype);
        let w=tensor(&data(e*n*k,5,dtype),[e,n,k],dtype);
        let g=tensor(&data(m*n,11,dtype),[m,n],dtype);
        let ids=from_data(TensorData::new((0..m).map(|i|(i/rows) as u32).collect::<Vec<_>>(),[m]),&Default::default());
        let offsets=from_data(TensorData::new((0..=e).map(|i|(i*rows) as u32).collect::<Vec<_>>(),[e+1]),&Default::default());
        let run=|s| {let z=unsafe {grouped_matmul_nt_backward_segmented_with_strategy(x.clone(),w.clone(),g.clone(),ids.clone(),offsets.clone(),s)}.unwrap();(floats(z.dinput),floats(z.dweights))};
        let expected=run(GroupedStrategy::Scalar);let actual=run(GroupedStrategy::TensorCore);
        close(&actual.0,&expected.0,0.008);close(&actual.1,&expected.1,0.008);
        for _ in 0..3 {run(GroupedStrategy::Scalar);run(GroupedStrategy::TensorCore);}
        for sample in 0..7 {let strategies=if sample%2==0 {[GroupedStrategy::Scalar,GroupedStrategy::TensorCore]} else {[GroupedStrategy::TensorCore,GroupedStrategy::Scalar]};
            for s in strategies {let start=Instant::now();let value=run(s);let us=start.elapsed().as_secs_f64()*1e6;
                close(&value.0,&expected.0,0.008);close(&value.1,&expected.1,0.008);
                println!("RUDA_V31_BENCH experts={e} rows_per_expert={rows} k={k} n={n} sample={sample} strategy={s:?} api_with_readback_us={us:.3}");
            }
        }
    }
}
