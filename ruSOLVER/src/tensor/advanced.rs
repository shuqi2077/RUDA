// SPDX-License-Identifier: Apache-2.0
//! Explicit opt-in device FP32 APIs. No casts, no implicit host computation.
use super::{DeviceSolverError,validate_tensor,elements,advanced_kernel};
use ruda_core::{device::Device,tensor::DType};
use ruda_kernel::{dsl::{Runtime,calculate_cube_count_elemwise,prelude::CubeDim},
    tensor::{RudaTensor,allocation::empty_device_contiguous_dtype,readback::into_data}};
const SOURCE:&str=concat!(include_str!("advanced_kernel.rs"),include_str!("advanced.rs"));
async fn status<R:Runtime>(info:RudaTensor<R>)->Result<(),DeviceSolverError>{
    let data=into_data(info).await.map_err(|e|DeviceSolverError::Readback(format!("{e:?}")))?;
    for(batch,&info)in data.to_vec::<i32>().map_err(|e|DeviceSolverError::Readback(format!("{e:?}")))?.iter().enumerate(){
        if info!=0{return Err(DeviceSolverError::MatrixFailure{batch,info});}
    }Ok(())
}
fn alloc<R:Runtime>(a:&RudaTensor<R>,shape:impl Into<ruda_core::tensor::Shape>,dtype:DType)->RudaTensor<R>{
    empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),shape.into(),dtype)
}
fn tolerance(atol:f32,rtol:f32,positive:bool)->Result<(),DeviceSolverError>{
    if !atol.is_finite()||!rtol.is_finite()||atol<0.0||rtol<0.0||(positive&&atol==0.0&&rtol==0.0){
        Err(DeviceSolverError::InvalidInput("finite nonnegative tolerances required"))}else{Ok(())}
}
fn square<R:Runtime>(a:&RudaTensor<R>,limit:usize)->Result<(usize,usize),DeviceSolverError>{
    validate_tensor(a)?;let s=a.meta.shape();if s[1]!=s[2]||s[1]==0||s[1]>limit{return Err(DeviceSolverError::InvalidInput("square order outside supported range"));}Ok((s[0],s[1]))
}
fn rhs<R:Runtime>(a:&RudaTensor<R>,b:&RudaTensor<R>,batch:usize,n:usize,max_rhs:usize)->Result<usize,DeviceSolverError>{
    validate_tensor(b)?;let s=b.meta.shape();if s[0]!=batch||s[1]!=n||s[2]==0||s[2]>max_rhs{return Err(DeviceSolverError::InvalidInput("RHS shape/range"));}
    if a.device.to_id()!=b.device.to_id()||!a.client.same_execution_queue(&b.client){return Err(DeviceSolverError::DifferentQueue);}Ok(s[2])
}
#[derive(Clone,Copy,Debug)]pub struct BatchedLuOptions{pub pivot_absolute_tolerance:f32,pub pivot_relative_tolerance:f32}
impl Default for BatchedLuOptions{fn default()->Self{Self{pivot_absolute_tolerance:0.0,pivot_relative_tolerance:1e-6}}}
pub struct BatchedLuResult<R:Runtime>{pub packed_lu:RudaTensor<R>,pub pivots:RudaTensor<R>,pub solution:RudaTensor<R>,pub info:RudaTensor<R>,pub submitted_kernels:usize}
impl<R:Runtime>BatchedLuResult<R>{pub fn check_status_sync(&self)->Result<(),DeviceSolverError>{ruda_core::future::block_on(status(self.info.clone()))}}
/// n=1..32, RHS=1..8. P A=L U with zero-based sequential row pivots. Positive
/// info is a singular pivot (one-based); -1 nonfinite input, -3 arithmetic failure.
/// Failed factors/solutions are zero and pivots -1. Input buffers are unchanged.
pub fn lu_solve_batched<R:Runtime>(a:&RudaTensor<R>,b:&RudaTensor<R>,o:BatchedLuOptions)->Result<BatchedLuResult<R>,DeviceSolverError>{
    let(batch,n)=square(a,32)?;let nr=rhs(a,b,batch,n,8)?;tolerance(o.pivot_absolute_tolerance,o.pivot_relative_tolerance,false)?;
    let packed_lu=alloc(a,[batch,n,n],DType::F32);let solution=alloc(a,[batch,n,nr],DType::F32);
    let pivots=alloc(a,[batch,n],DType::I32);let info=alloc(a,[batch],DType::I32);
    if batch>0{let dim=CubeDim::new(a.client.properties(),batch);
        advanced_kernel::lu_solve::launch::<R>(&a.client,calculate_cube_count_elemwise(&a.client,batch,dim),dim,
            a.clone().into_array_arg(),b.clone().into_array_arg(),packed_lu.clone().into_array_arg(),solution.clone().into_array_arg(),
            pivots.clone().into_array_arg(),info.clone().into_array_arg(),n as u32,nr as u32,o.pivot_absolute_tolerance,o.pivot_relative_tolerance,SOURCE.to_owned());}
    Ok(BatchedLuResult{packed_lu,pivots,solution,info,submitted_kernels:usize::from(batch>0)})
}
#[derive(Clone,Copy,Debug)]pub struct BatchedQrOptions{pub rank_absolute_tolerance:f32,pub rank_relative_tolerance:f32}
impl Default for BatchedQrOptions{fn default()->Self{Self{rank_absolute_tolerance:0.0,rank_relative_tolerance:1e-6}}}
pub struct BatchedQrResult<R:Runtime>{pub q:RudaTensor<R>,pub r:RudaTensor<R>,pub info:RudaTensor<R>,pub submitted_kernels:usize}
impl<R:Runtime>BatchedQrResult<R>{pub fn check_status_sync(&self)->Result<(),DeviceSolverError>{ruda_core::future::block_on(status(self.info.clone()))}}
/// Unpivoted Householder thin QR: 1<=n<=32, n<=m<=64. Positive info denotes a
/// numerical rank failure, -1 nonfinite input, -3 arithmetic. No least-squares API.
pub fn qr_batched<R:Runtime>(a:&RudaTensor<R>,o:BatchedQrOptions)->Result<BatchedQrResult<R>,DeviceSolverError>{
    validate_tensor(a)?;let s=a.meta.shape();let(batch,m,n)=(s[0],s[1],s[2]);
    if n==0||n>32||m<n||m>64{return Err(DeviceSolverError::InvalidInput("QR requires 1<=n<=32, n<=m<=64"));}
    tolerance(o.rank_absolute_tolerance,o.rank_relative_tolerance,false)?;
    let q=alloc(a,[batch,m,n],DType::F32);let r=alloc(a,[batch,n,n],DType::F32);let info=alloc(a,[batch],DType::I32);
    if batch>0{let work=alloc(a,[batch,m,n],DType::F32);let tau=alloc(a,[batch,n],DType::F32);let dim=CubeDim::new(a.client.properties(),batch);
        advanced_kernel::qr::launch::<R>(&a.client,calculate_cube_count_elemwise(&a.client,batch,dim),dim,a.clone().into_array_arg(),q.clone().into_array_arg(),
            r.clone().into_array_arg(),work.into_array_arg(),tau.into_array_arg(),info.clone().into_array_arg(),m as u32,n as u32,
            o.rank_absolute_tolerance,o.rank_relative_tolerance,SOURCE.to_owned());}
    Ok(BatchedQrResult{q,r,info,submitted_kernels:usize::from(batch>0)})
}
#[derive(Clone,Copy,Debug)]pub struct BatchedEigenOptions{pub absolute_tolerance:f32,pub relative_tolerance:f32,pub symmetry_tolerance:f32,pub max_sweeps:u32}
impl Default for BatchedEigenOptions{fn default()->Self{Self{absolute_tolerance:0.0,relative_tolerance:1e-6,symmetry_tolerance:1e-5,max_sweeps:64}}}
pub struct BatchedEigenResult<R:Runtime>{pub values:RudaTensor<R>,pub vectors:RudaTensor<R>,pub sweeps:RudaTensor<R>,pub info:RudaTensor<R>,pub submitted_kernels:usize}
impl<R:Runtime>BatchedEigenResult<R>{pub fn check_status_sync(&self)->Result<(),DeviceSolverError>{ruda_core::future::block_on(status(self.info.clone()))}}
/// Symmetric eigenproblem, n=1..32. Ascending eigenvalues, columns are vectors.
/// The lower triangle defines the accepted symmetric matrix. -2 nonsymmetry,
/// -4 nonconvergence; failures zero outputs. Tolerance uses scaled Frobenius norm.
pub fn symmetric_eigen_batched<R:Runtime>(a:&RudaTensor<R>,o:BatchedEigenOptions)->Result<BatchedEigenResult<R>,DeviceSolverError>{
    let(batch,n)=square(a,32)?;tolerance(o.absolute_tolerance,o.relative_tolerance,true)?;tolerance(0.0,o.symmetry_tolerance,false)?;
    if o.max_sweeps==0||o.max_sweeps>256{return Err(DeviceSolverError::InvalidInput("eigen sweeps 1..256"));}
    let values=alloc(a,[batch,n],DType::F32);let vectors=alloc(a,[batch,n,n],DType::F32);let info=alloc(a,[batch],DType::I32);let sweeps=alloc(a,[batch],DType::I32);
    if batch>0{let work=alloc(a,[batch,n,n],DType::F32);let dim=CubeDim::new(a.client.properties(),batch);
        advanced_kernel::eigen::launch::<R>(&a.client,calculate_cube_count_elemwise(&a.client,batch,dim),dim,a.clone().into_array_arg(),values.clone().into_array_arg(),
            vectors.clone().into_array_arg(),work.into_array_arg(),info.clone().into_array_arg(),sweeps.clone().into_array_arg(),n as u32,o.max_sweeps,
            o.absolute_tolerance,o.relative_tolerance,o.symmetry_tolerance,SOURCE.to_owned());}
    Ok(BatchedEigenResult{values,vectors,sweeps,info,submitted_kernels:usize::from(batch>0)})
}
#[derive(Clone,Copy,Debug)]pub struct BatchedCgOptions{pub absolute_tolerance:f32,pub relative_tolerance:f32,pub symmetry_tolerance:f32,pub max_iterations:u32,pub jacobi:bool}
impl Default for BatchedCgOptions{fn default()->Self{Self{absolute_tolerance:0.0,relative_tolerance:1e-5,symmetry_tolerance:1e-5,max_iterations:512,jacobi:true}}}
pub struct BatchedCgResult<R:Runtime>{pub solution:RudaTensor<R>,pub residual_norm:RudaTensor<R>,pub iterations:RudaTensor<R>,pub info:RudaTensor<R>,pub submitted_kernels:usize}
impl<R:Runtime>BatchedCgResult<R>{pub fn check_status_sync(&self)->Result<(),DeviceSolverError>{ruda_core::future::block_on(status(self.info.clone()))}}
/// Dense SPD CG, 1<=n<=128, one RHS [batch,n,1], zero initial guess. Full matrix
/// stays on device. Each independent system has its own iterations/convergence.
/// -4 returns an unconverged APPROXIMATION (not a successful solution); -5 means
/// curvature/preconditioner breakdown. Other errors zero outputs. Status mandatory.
pub fn conjugate_gradient_batched<R:Runtime>(a:&RudaTensor<R>,b:&RudaTensor<R>,o:BatchedCgOptions)->Result<BatchedCgResult<R>,DeviceSolverError>{
    let(batch,n)=square(a,128)?;rhs(a,b,batch,n,1)?;tolerance(o.absolute_tolerance,o.relative_tolerance,true)?;tolerance(0.0,o.symmetry_tolerance,false)?;
    if o.max_iterations>4096{return Err(DeviceSolverError::InvalidInput("CG iteration budget <=4096"));}
    elements(&[batch,4,n])?;
    let solution=alloc(a,[batch,n,1],DType::F32);let info=alloc(a,[batch],DType::I32);let residual_norm=alloc(a,[batch],DType::F32);let iterations=alloc(a,[batch],DType::I32);
    if batch>0{let scratch=alloc(a,[batch,4,n],DType::F32);let dim=CubeDim::new(a.client.properties(),batch);
        advanced_kernel::cg::launch::<R>(&a.client,calculate_cube_count_elemwise(&a.client,batch,dim),dim,a.clone().into_array_arg(),b.clone().into_array_arg(),solution.clone().into_array_arg(),
            scratch.into_array_arg(),info.clone().into_array_arg(),iterations.clone().into_array_arg(),residual_norm.clone().into_array_arg(),n as u32,o.max_iterations,
            o.absolute_tolerance,o.relative_tolerance,o.symmetry_tolerance,u32::from(o.jacobi),SOURCE.to_owned());}
    Ok(BatchedCgResult{solution,residual_norm,iterations,info,submitted_kernels:usize::from(batch>0)})
}
