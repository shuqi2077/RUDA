// SPDX-License-Identifier: Apache-2.0
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{
    CudaDevice, CudaRuntime
};
use ruda_kernel::tensor::{
    transfer::from_data, readback::into_data_sync
};
use rusolver::tensor::cholesky_solve_batched;
fn main()->Result<(), Box<dyn std::error::Error>>{
    let device=CudaDevice::default();
    let a=from_data::<CudaRuntime>(TensorData::new(vec![4.0f32, 1.0, 1.0, 3.0, 9.0, 0.0, 0.0, 4.0], [2, 2, 2]), &device);
    let b=from_data::<CudaRuntime>(TensorData::new(vec![6.0f32, 7.0, 18.0, 12.0], [2, 2, 1]), &device);
    let result=cholesky_solve_batched(&a, &b, Default::default())?;
    result.check_status_sync()?;
    let values=into_data_sync(result.solution).to_vec::<f32>()?;
    for(&got, want)in values.iter().zip([1.0, 2.0, 2.0, 3.0]){
        assert!((got-want).abs()<1e-5);
    }
    println!("native FP32 batched Cholesky solve: {values:?}");
    Ok(())
}
