// SPDX-License-Identifier: Apache-2.0
//! Requires an actual CUDA device; do not skip and report success when unavailable.
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice,CudaRuntime};
use ruda_kernel::tensor::{RudaTensor,transfer::from_data,readback::into_data_sync};
use rusolver::{Matrix,Lu,symmetric_eigen};
use rusolver::tensor::*;
fn upload(v:Vec<f32>,s:[usize;3])->RudaTensor<CudaRuntime>{from_data(TensorData::new(v,s),&CudaDevice::default())}
fn read(t:RudaTensor<CudaRuntime>)->Vec<f32>{into_data_sync(t).to_vec::<f32>().unwrap()}
fn codes(t:RudaTensor<CudaRuntime>)->Vec<i32>{into_data_sync(t).to_vec::<i32>().unwrap()}
fn close(x:&[f32],y:&[f64],tol:f64){assert_eq!(x.len(),y.len());for(&x,&y)in x.iter().zip(y){assert!((x as f64-y).abs()<=tol*(1.+y.abs()),"{x} vs {y}");}}
#[test]fn gpu_lu_pivot_tails_and_rhs(){let a0=[0.,2.,1.,1.,-2.,-3.,3.,-1.,2.];let b0=[1.,2.,3.,4.,5.,6.];for batch in[1,3,65]{
    let av=a0.repeat(batch);let bv=b0.repeat(batch);let a=upload(av.clone(),[batch,3,3]);let b=upload(bv.clone(),[batch,3,2]);
    let r=lu_solve_batched(&a,&b,Default::default()).unwrap();r.check_status_sync().unwrap();assert_eq!(r.submitted_kernels,1);
    let expected=Lu::factor(Matrix::from_f32(3,3,&a0).unwrap().view(),Default::default()).unwrap().solve(Matrix::from_f32(3,2,&b0).unwrap().view()).unwrap();
    close(&read(r.solution),&expected.values().repeat(batch),5e-5);assert_eq!(read(a),av);assert_eq!(read(b),bv);
}}
#[test]fn gpu_lu_status_and_no_input_change(){let a=upload(vec![1.,2.,2.,4.,1.,0.,0.,f32::NAN],[2,2,2]);let b=upload(vec![1.;4],[2,2,1]);let r=lu_solve_batched(&a,&b,Default::default()).unwrap();assert!(r.check_status_sync().is_err());let c=codes(r.info);assert!(c[0]>0);assert_eq!(c[1],-1);assert!(read(r.solution).iter().all(|x|*x==0.));}
#[test]fn gpu_qr_reconstruction_and_orthogonality(){for(m,n,batch)in[(4,2,3),(16,8,2),(32,16,1)]{let mut a=vec![0.;batch*m*n];for s in 0..batch{for i in 0..m{for j in 0..n{a[s*m*n+i*n+j]=((i*7+j*3+s)%17)as f32/17.+if i==j{3.}else{0.};}}}
    let r=qr_batched(&upload(a.clone(),[batch,m,n]),Default::default()).unwrap();r.check_status_sync().unwrap();let q=read(r.q);let rr=read(r.r);
    for s in 0..batch{for i in 0..m{for j in 0..n{let z=(0..n).map(|k|q[s*m*n+i*n+k]as f64*rr[s*n*n+k*n+j]as f64).sum::<f64>();assert!((z-a[s*m*n+i*n+j]as f64).abs()<2e-4);}}
        for i in 0..n{for j in 0..n{let z=(0..m).map(|k|q[s*m*n+k*n+i]as f64*q[s*m*n+k*n+j]as f64).sum::<f64>();assert!((z-if i==j{1.}else{0.}).abs()<2e-4);}}}
}}
#[test]fn gpu_qr_rank_failure(){let a=upload(vec![1.,2.,2.,4.,3.,6.],[1,3,2]);let r=qr_batched(&a,Default::default()).unwrap();assert!(r.check_status_sync().is_err());assert!(codes(r.info)[0]>0);}
#[test]fn gpu_eigen_values_and_residual(){for n in[1,3,8,16]{let mut a=vec![0f32;n*n];for i in 0..n{for j in 0..n{a[i*n+j]=if i==j{(i+3)as f32}else{0.1*((i+j)%3)as f32};}}
    let r=symmetric_eigen_batched(&upload(a.clone(),[1,n,n]),Default::default()).unwrap();r.check_status_sync().unwrap();let vals=read(r.values);let v=read(r.vectors);
    let expected=symmetric_eigen(Matrix::from_f32(n,n,&a).unwrap().view(),Default::default()).unwrap();close(&vals,&expected.values,1e-4);
    for i in 0..n{for j in 0..n{let z=(0..n).map(|k|a[i*n+k]as f64*v[k*n+j]as f64).sum::<f64>();assert!((z-vals[j]as f64*v[i*n+j]as f64).abs()<3e-4);}}
}}
#[test]fn gpu_eigen_nonsymmetry_and_nonconvergence(){let r=symmetric_eigen_batched(&upload(vec![1.,2.,0.,1.],[1,2,2]),Default::default()).unwrap();assert_eq!(codes(r.info),vec![-2]);
    let a=upload(vec![4.,1.,2.,1.,3.,0.5,2.,0.5,5.],[1,3,3]);let r=symmetric_eigen_batched(&a,BatchedEigenOptions{relative_tolerance:1e-12,max_sweeps:1,..Default::default()}).unwrap();assert_eq!(codes(r.info),vec![-4]);}
#[test]fn gpu_cg_dense_true_residual(){for n in[1,7,32,128]{let batch=3;let mut av=vec![0.;batch*n*n];let mut bv=vec![0.;batch*n];for s in 0..batch{for i in 0..n{av[s*n*n+i*n+i]=3.;bv[s*n+i]=3.;if i>0{av[s*n*n+i*n+i-1]=-1.;bv[s*n+i]-=1.;}if i+1<n{av[s*n*n+i*n+i+1]=-1.;bv[s*n+i]-=1.;}}}
    let r=conjugate_gradient_batched(&upload(av,[batch,n,n]),&upload(bv,[batch,n,1]),Default::default()).unwrap();r.check_status_sync().unwrap();close(&read(r.solution),&vec![1.;batch*n],5e-4);
}}
#[test]fn gpu_cg_zero_rhs_budget_and_breakdown(){let a=upload(vec![4.,1.,1.,3.],[1,2,2]);let b=upload(vec![0.,0.],[1,2,1]);let r=conjugate_gradient_batched(&a,&b,Default::default()).unwrap();r.check_status_sync().unwrap();assert_eq!(codes(r.iterations),vec![0]);
    let b=upload(vec![1.,2.],[1,2,1]);let r=conjugate_gradient_batched(&a,&b,BatchedCgOptions{max_iterations:0,..Default::default()}).unwrap();assert_eq!(codes(r.info),vec![-4]);
    let bad=upload(vec![-1.,0.,0.,1.],[1,2,2]);let r=conjugate_gradient_batched(&bad,&b,Default::default()).unwrap();assert_eq!(codes(r.info),vec![-5]);}
#[test]fn all_gpu_empty_batches_and_bad_options(){let a=upload(vec![],[0,2,2]);let b=upload(vec![],[0,2,1]);assert_eq!(lu_solve_batched(&a,&b,Default::default()).unwrap().submitted_kernels,0);
    assert_eq!(qr_batched(&a,Default::default()).unwrap().submitted_kernels,0);assert_eq!(symmetric_eigen_batched(&a,Default::default()).unwrap().submitted_kernels,0);
    assert_eq!(conjugate_gradient_batched(&a,&b,Default::default()).unwrap().submitted_kernels,0);assert!(symmetric_eigen_batched(&a,BatchedEigenOptions{max_sweeps:0,..Default::default()}).is_err());}
