use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use rusparse::{
    CsrMatrix, IndexBase, Operation,
    tensor::{CsrTensor, csrmv},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let host = CsrMatrix::new(
        3,
        3,
        &[0, 2, 3, 4],
        &[0, 1, 1, 2],
        &[2., 1., -1., 5.],
        IndexBase::Zero,
    )?;
    let matrix = CsrTensor::<CudaRuntime>::from_csr(host, Operation::None, &device)?;
    let x = from_data(TensorData::new(vec![3f32, 4., 5.], [3]), &device);
    let y = from_data(TensorData::new(vec![1f32, 2., 3.], [3]), &device);
    let output = into_data_sync(csrmv(&matrix, 0.5, x, 2., y)?).to_vec::<f32>()?;
    assert_eq!(output, [7., 2., 18.5]);
    println!("alpha * A * x + beta * y = {output:?}");
    Ok(())
}
