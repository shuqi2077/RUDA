// SPDX-License-Identifier: Apache-2.0
//! Real CUDA smoke workload; missing/unsupported devices fail explicitly.
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{RudaTensor, transfer::from_data, readback::into_data_sync};
use rusolver::tensor::*;
fn upload(v: Vec<f32>, shape: [usize; 3]) -> RudaTensor<CudaRuntime> {
    from_data(TensorData::new(v, shape), &CudaDevice::default())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = upload(vec![4.0, 1.0, 1.0, 3.0], [1, 2, 2]);
    let b = upload(vec![6.0, 7.0], [1, 2, 1]);
    let lu = lu_solve_batched(&a, &b, Default::default())?;
    lu.check_status_sync()?;
    let x = into_data_sync(lu.solution).to_vec::<f32>()?;
    assert!((x[0]-1.0).abs()<1e-5 && (x[1]-2.0).abs()<1e-5);
    let qr = qr_batched(&a, Default::default())?;
    qr.check_status_sync()?;
    let eig = symmetric_eigen_batched(&a, Default::default())?;
    eig.check_status_sync()?;
    let cg = conjugate_gradient_batched(&a, &b, Default::default())?;
    cg.check_status_sync()?;
    let x = into_data_sync(cg.solution).to_vec::<f32>()?;
    assert!((x[0]-1.0).abs()<1e-4 && (x[1]-2.0).abs()<1e-4);
    println!("LU, QR, symmetric eigen, CG completed; LU/CG solution checked. No performance claim.");
    Ok(())
}
