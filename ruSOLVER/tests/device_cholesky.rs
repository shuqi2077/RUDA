// SPDX-License-Identifier: Apache-2.0
//! Hardware-required tests: explicitly selected, never reported as pass without CUDA.
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{
    CudaDevice, CudaRuntime
};
use ruda_kernel::tensor::{
    RudaTensor, transfer::from_data, readback::into_data_sync
};
use rusolver::{
    Matrix, Cholesky, tensor::{
        cholesky_solve_batched, BatchedCholeskyOptions, DeviceSolverError
    }
};
fn upload(values: Vec<f32>, shape: [usize; 3])->RudaTensor<CudaRuntime>{
    from_data(TensorData::new(values, shape), &CudaDevice::default())
}
fn read(t: RudaTensor<CudaRuntime>)->Vec<f32>{
    into_data_sync(t).to_vec::<f32>().unwrap()
}
#[test]
fn batches_tails_and_multiple_rhs_match_independent_host_factors(){
    for (batch, n, nrhs) in [(1, 1, 1), (3, 2, 3), (65, 7, 1), (5, 16, 4), (2, 32, 8)]{
        let mut av=vec![0.0f32; batch*n*n];
        let mut bv=vec![0.0f32; batch*n*nrhs];
        let mut expected=Vec::new();
        for s in 0..batch{
            let raw: Vec<f64>=(0..n*n).map(|i|((i*7+s*3)%19) as f64/19.0-0.5).collect();
            for i in 0..n{
                for j in 0..n{
                    let mut v=if i==j{
                        n as f64
                    } else{
                        0.0
                    };
                    for k in 0..n{
                        v+=raw[i*n+k]*raw[j*n+k];
                    }
                    av[s*n*n+i*n+j]=v as f32;
                }
            }
            for i in 0..n*nrhs{
                bv[s*n*nrhs+i]=((i+s)%23) as f32/11.0-1.0;
            }
            let a=Matrix::from_f32(n, n, &av[s*n*n..(s+1)*n*n]).unwrap();
            let b=Matrix::from_f32(n, nrhs, &bv[s*n*nrhs..(s+1)*n*nrhs]).unwrap();
            expected.extend(Cholesky::factor(a.view(), Default::default()).unwrap().solve(b.view()).unwrap().into_values());
        }
        let a=upload(av.clone(), [batch, n, n]);
        let b=upload(bv.clone(), [batch, n, nrhs]);
        let result=cholesky_solve_batched(&a, &b, Default::default()).unwrap();
        assert_eq!(result.submitted_kernels, 1);
        result.check_status_sync().unwrap();
        let got=read(result.solution);
        for(i, (&got, &want))in got.iter().zip(&expected).enumerate(){
            assert!((got as f64-want).abs()<3e-5*(1.0+want.abs()), "{i}: {got} != {want}");
        }
        assert_eq!(read(a), av);
        assert_eq!(read(b), bv);
    }
}
#[test]
fn mixed_invalid_systems_report_status_and_zero_outputs(){
    let a=upload(vec![4.0, 1.0, 1.0, 3.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0, f32::NAN, 0.0, 0.0, 1.0], [4, 2, 2]);
    let b=upload(vec![1.0; 8], [4, 2, 1]);
    let r=cholesky_solve_batched(&a, &b, Default::default()).unwrap();
    assert!(matches!(r.check_status_sync(), Err(DeviceSolverError::MatrixFailure{
        batch: 1, info: 2
    })));
    assert_eq!(into_data_sync(r.info).to_vec::<i32>().unwrap(), [0, 2, -2, -1]);
    assert!(read(r.solution)[2..].iter().all(|x|*x==0.0));
    assert!(read(r.lower)[4..].iter().all(|x|*x==0.0));
}
#[test]
fn explicit_diagonal_shift_and_nonfinite_rhs(){
    let a=upload(vec![0.0, 0.0, 0.0, 0.0], [1, 2, 2]);
    let b=upload(vec![2.0, 4.0], [1, 2, 1]);
    let r=cholesky_solve_batched(&a, &b, BatchedCholeskyOptions{
        diagonal_shift: 2.0, ..Default::default()
    }).unwrap();
    r.check_status_sync().unwrap();
    let x=read(r.solution);
    assert!((x[0]-1.0).abs()<1e-5&&(x[1]-2.0).abs()<1e-5);
    let bad=upload(vec![f32::INFINITY, 0.0], [1, 2, 1]);
    let r=cholesky_solve_batched(&a, &bad, Default::default()).unwrap();
    assert!(matches!(r.check_status_sync(), Err(DeviceSolverError::MatrixFailure{
        info: -1, ..
    })));
}
#[test]
fn malformed_and_strided_inputs_are_rejected_before_launch(){
    let a=upload(vec![1.0; 9], [1, 3, 3]);
    let b=upload(vec![1.0; 3], [1, 3, 1]);
    let mut transpose=a.clone();
    transpose.meta.swap(1, 2);
    assert!(cholesky_solve_batched(&transpose, &b, Default::default()).is_err());
    let big=upload(vec![1.0; 33*33], [1, 33, 33]);
    let bb=upload(vec![1.0; 33], [1, 33, 1]);
    assert!(cholesky_solve_batched(&big, &bb, Default::default()).is_err());
    assert!(cholesky_solve_batched(&a, &b, BatchedCholeskyOptions{
        diagonal_shift: -1.0, ..Default::default()
    }).is_err());
}
#[test]
fn empty_batch_submits_no_kernel(){
    let a=upload(vec![], [0, 2, 2]);
    let b=upload(vec![], [0, 2, 1]);
    let r=cholesky_solve_batched(&a, &b, Default::default()).unwrap();
    assert_eq!(r.submitted_kernels, 0);
}
