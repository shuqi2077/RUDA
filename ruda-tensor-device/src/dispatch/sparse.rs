use crate::{DeviceBackend, DeviceRuntime, FloatElement, IntElement, element::BoolElement};
use ruda_tensor::{DType, TensorMetadata, ops::{CsrAddition, SparseOps}, tensor::FloatTensor};
use ruda_core::device::Device;
use ruda_kernel::tensor::initialization::zeros;
use rusparse::{DenseOrder, Operation, SparseError, tensor::{CsrTensor, csrmm, sddmm}};

impl<R, F, I, BT> SparseOps for DeviceBackend<R, F, I, BT>
where
    R: DeviceRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    type CsrHandle = CsrTensor<R>;
    type CsrData = rusparse::CsrMatrixOwned;
    type SparseError = SparseError;

    fn csr_from_data(data: &Self::CsrData, device: &Self::Device) -> Result<Self::CsrHandle, SparseError> {
        CsrTensor::from_csr(data.as_ref(), Operation::None, device)
    }

    fn csr_to_device(matrix: &Self::CsrHandle, device: &Self::Device) -> Self::CsrHandle {
        matrix.to_device(device)
    }

    fn csr_transpose_with_permutation(matrix: &Self::CsrHandle) -> Result<(Self::CsrHandle, Vec<u32>), SparseError> {
        matrix.transpose_with_permutation()
    }

    async fn csr_to_data(matrix: &Self::CsrHandle, values: FloatTensor<Self>) -> Result<Self::CsrData, SparseError> {
        matrix.with_values(values)?.to_csr().await
    }

    fn csr_shape(matrix: &Self::CsrHandle) -> [usize; 2] {
        [matrix.rows(), matrix.columns()]
    }

    fn csr_nnz(matrix: &Self::CsrHandle) -> usize {
        matrix.nnz()
    }

    fn csr_product_pattern(left: &Self::CsrHandle, right: &Self::CsrHandle) -> Result<Self::CsrHandle, SparseError> {
        left.product_pattern(right)
    }

    fn csr_sampled_sparse_matmul(
        pattern: &Self::CsrHandle, left: &Self::CsrHandle, left_values: FloatTensor<Self>,
        right: &Self::CsrHandle, right_values: FloatTensor<Self>, transpose_left: bool, transpose_right: bool,
    ) -> Result<FloatTensor<Self>, SparseError> {
        let left = left.with_values(left_values)?;
        let right = right.with_values(right_values)?;
        let operation = |transpose| if transpose { Operation::Transpose } else { Operation::None };
        rusparse::tensor::sampled_csrgemm(pattern, operation(transpose_left), operation(transpose_right), &left, &right)
    }

    fn csr_add_prepare(left: &Self::CsrHandle, right: &Self::CsrHandle) -> Result<CsrAddition<Self::CsrHandle>, SparseError> {
        let (output, left_entries, right_entries) = left.sum_pattern(right)?;
        Ok(CsrAddition { left: left.clone(), right: right.clone(), output, left_entries, right_entries })
    }

    fn csr_add(plan: &CsrAddition<Self::CsrHandle>, left: FloatTensor<Self>, right: FloatTensor<Self>, alpha: f32, beta: f32) -> Result<FloatTensor<Self>, SparseError> {
        let left = plan.left.with_values(left)?;
        let right = plan.right.with_values(right)?;
        rusparse::tensor::csrgeam(Operation::None, Operation::None, alpha, &left, beta, &right)
            .map(|matrix| matrix.values())
    }

    fn csr_validate_operand<T: TensorMetadata>(
        matrix: &Self::CsrHandle,
        operand: &T,
        device: &Self::Device,
        shape: &[usize],
    ) -> Result<(), SparseError> {
        if operand.shape()[..] != *shape {
            return Err(SparseError::DimensionMismatch("sparse dense operand shape mismatch"));
        }
        if operand.dtype() != DType::F32 || device.to_id() != matrix.device().to_id() {
            return Err(SparseError::Device("sparse operands must be same-device unquantized F32 tensors"));
        }
        Ok(())
    }

    fn csr_values(matrix: &Self::CsrHandle) -> FloatTensor<Self> {
        matrix.values()
    }

    fn csr_validate_values(matrix: &Self::CsrHandle, values: &FloatTensor<Self>) -> Result<(), SparseError> {
        matrix.with_values(values.clone()).map(|_| ())
    }

    fn csr_gather(matrix: &Self::CsrHandle, dense: FloatTensor<Self>) -> Result<FloatTensor<Self>, SparseError> {
        rusparse::tensor::csr_gather(matrix, dense)
    }

    fn csr_scatter_add(matrix: &Self::CsrHandle, values: FloatTensor<Self>) -> Result<FloatTensor<Self>, SparseError> {
        rusparse::tensor::csr_scatter_add(&matrix.with_values(values)?)
    }

    fn csr_to_dense(matrix: &Self::CsrHandle, values: FloatTensor<Self>) -> Result<FloatTensor<Self>, SparseError> {
        let matrix = matrix.with_values(values)?;
        rusparse::tensor::csr_to_dense(&matrix, DenseOrder::RowMajor)
    }

    fn csr_to_dense_backward(matrix: &Self::CsrHandle, grad: FloatTensor<Self>) -> Result<FloatTensor<Self>, SparseError> {
        rusparse::tensor::csr_to_dense_backward(matrix, grad)
    }

    fn csr_matmul(
        matrix: &Self::CsrHandle,
        values: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        transpose: bool,
    ) -> Result<FloatTensor<Self>, SparseError> {
        let matrix = matrix.with_values(values)?;
        let matrix = if transpose { matrix.transpose()? } else { matrix };
        csrmm(&matrix, Operation::None, 1.0, rhs, 0.0, None, DenseOrder::RowMajor)
    }

    fn csr_sampled_matmul(
        matrix: &Self::CsrHandle,
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
    ) -> Result<FloatTensor<Self>, SparseError> {
        let zero = zeros::<R>(matrix.device().clone(), [matrix.nnz()].into(), DType::F32);
        let matrix = matrix.with_values(zero)?;
        sddmm(Operation::None, Operation::None, 1.0, lhs, rhs, 0.0, &matrix)
            .map(|output| output.values())
    }
}
