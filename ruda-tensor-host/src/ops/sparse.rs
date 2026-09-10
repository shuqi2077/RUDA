use alloc::{sync::Arc, vec::Vec};
use core::fmt::{Display, Formatter};
use ruda_tensor::{DType, DataError, TensorData, TensorMetadata, ops::{CsrAddition, SparseOps}};
use rusparse::{CsrMatrix, CsrMatrixOwned, DenseMatrix, DenseOrder, SparseError};

use crate::{Host, HostDevice, HostTensor};

#[derive(Debug)]
pub enum HostSparseError {
    Sparse(SparseError),
    Data(DataError),
}

impl From<SparseError> for HostSparseError {
    fn from(error: SparseError) -> Self { Self::Sparse(error) }
}

impl From<DataError> for HostSparseError {
    fn from(error: DataError) -> Self { Self::Data(error) }
}

impl Display for HostSparseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Sparse(error) => Display::fmt(error, formatter),
            Self::Data(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for HostSparseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sparse(error) => Some(error),
            Self::Data(error) => Some(error),
        }
    }
}

impl SparseOps for Host {
    type CsrHandle = Arc<CsrMatrixOwned>;
    type CsrData = CsrMatrixOwned;
    type SparseError = HostSparseError;

    fn csr_from_data(data: &CsrMatrixOwned, _device: &HostDevice) -> Result<Self::CsrHandle, HostSparseError> {
        Ok(Arc::new(data.clone()))
    }

    fn csr_to_device(matrix: &Self::CsrHandle, _device: &HostDevice) -> Self::CsrHandle {
        matrix.clone()
    }

    fn csr_transpose_with_permutation(matrix: &Self::CsrHandle) -> Result<(Self::CsrHandle, Vec<u32>), HostSparseError> {
        let (matrix, permutation) = matrix.as_ref().as_ref().transpose_with_permutation()?;
        Ok((Arc::new(matrix), permutation))
    }

    async fn csr_to_data(matrix: &Self::CsrHandle, values: HostTensor) -> Result<CsrMatrixOwned, HostSparseError> {
        Self::csr_validate_values(matrix, &values)?;
        let pattern = matrix.as_ref().as_ref();
        Ok(CsrMatrixOwned::new(
            pattern.rows(), pattern.columns(), pattern.row_offsets().to_vec(),
            pattern.column_indices().to_vec(), tensor_values(values)?, pattern.index_base(),
        )?)
    }

    fn csr_shape(matrix: &Self::CsrHandle) -> [usize; 2] {
        let matrix = matrix.as_ref().as_ref();
        [matrix.rows(), matrix.columns()]
    }

    fn csr_nnz(matrix: &Self::CsrHandle) -> usize {
        matrix.as_ref().as_ref().nnz()
    }

    fn csr_product_pattern(left: &Self::CsrHandle, right: &Self::CsrHandle) -> Result<Self::CsrHandle, HostSparseError> {
        Ok(Arc::new(rusparse::host::csr_product_pattern(left.as_ref().as_ref(), right.as_ref().as_ref())?))
    }

    fn csr_sampled_sparse_matmul(
        pattern: &Self::CsrHandle, left: &Self::CsrHandle, left_values: HostTensor,
        right: &Self::CsrHandle, right_values: HostTensor, transpose_left: bool, transpose_right: bool,
    ) -> Result<HostTensor, HostSparseError> {
        Self::csr_validate_values(left, &left_values)?;
        Self::csr_validate_values(right, &right_values)?;
        let left_values = tensor_values(left_values)?;
        let right_values = tensor_values(right_values)?;
        let left = left.as_ref().as_ref();
        let right = right.as_ref().as_ref();
        let left = CsrMatrix::new(left.rows(), left.columns(), left.row_offsets(), left.column_indices(), &left_values, left.index_base())?;
        let right = CsrMatrix::new(right.rows(), right.columns(), right.row_offsets(), right.column_indices(), &right_values, right.index_base())?;
        let operation = |transpose| if transpose { rusparse::Operation::Transpose } else { rusparse::Operation::None };
        let values = rusparse::host::sampled_csrgemm(pattern.as_ref().as_ref(), operation(transpose_left), operation(transpose_right), left, right)?;
        Ok(HostTensor::from_data(TensorData::new(values, [Self::csr_nnz(pattern)])))
    }

    fn csr_add_prepare(left: &Self::CsrHandle, right: &Self::CsrHandle) -> Result<CsrAddition<Self::CsrHandle>, HostSparseError> {
        let (output, left_entries, right_entries) = rusparse::host::csr_sum_pattern(left.as_ref().as_ref(), right.as_ref().as_ref())?;
        Ok(CsrAddition { left: left.clone(), right: right.clone(), output: Arc::new(output), left_entries, right_entries })
    }

    fn csr_add(plan: &CsrAddition<Self::CsrHandle>, left: HostTensor, right: HostTensor, alpha: f32, beta: f32) -> Result<HostTensor, HostSparseError> {
        Self::csr_validate_values(&plan.left, &left)?;
        Self::csr_validate_values(&plan.right, &right)?;
        let left_values = tensor_values(left)?;
        let right_values = tensor_values(right)?;
        let left_pattern = plan.left.as_ref().as_ref();
        let right_pattern = plan.right.as_ref().as_ref();
        let left = CsrMatrix::new(left_pattern.rows(), left_pattern.columns(), left_pattern.row_offsets(), left_pattern.column_indices(), &left_values, left_pattern.index_base())?;
        let right = CsrMatrix::new(right_pattern.rows(), right_pattern.columns(), right_pattern.row_offsets(), right_pattern.column_indices(), &right_values, right_pattern.index_base())?;
        let output = rusparse::host::csrgeam(rusparse::Operation::None, rusparse::Operation::None, alpha, left, beta, right)?;
        let output = output.as_ref();
        Ok(HostTensor::from_data(TensorData::new(output.values().to_vec(), [output.nnz()])))
    }

    fn csr_validate_operand<T: TensorMetadata>(
        _matrix: &Self::CsrHandle,
        operand: &T,
        _device: &HostDevice,
        shape: &[usize],
    ) -> Result<(), HostSparseError> {
        if operand.shape()[..] != *shape {
            return Err(SparseError::DimensionMismatch("sparse dense operand shape mismatch").into());
        }
        if operand.dtype() != DType::F32 {
            return Err(SparseError::Device("host sparse operands must be unquantized F32 tensors").into());
        }
        Ok(())
    }

    fn csr_values(matrix: &Self::CsrHandle) -> HostTensor {
        let matrix = matrix.as_ref().as_ref();
        HostTensor::from_data(TensorData::new(matrix.values().to_vec(), [matrix.nnz()]))
    }

    fn csr_validate_values(matrix: &Self::CsrHandle, values: &HostTensor) -> Result<(), HostSparseError> {
        Self::csr_validate_operand(matrix, values, &HostDevice, &[Self::csr_nnz(matrix)])
    }

    fn csr_gather(matrix: &Self::CsrHandle, dense: HostTensor) -> Result<HostTensor, HostSparseError> {
        let pattern = matrix.as_ref().as_ref();
        Self::csr_validate_operand(matrix, &dense, &HostDevice, &[pattern.rows(), pattern.columns()])?;
        let values = tensor_values(dense)?;
        let dense = DenseMatrix::new(&values, pattern.rows(), pattern.columns(), DenseOrder::RowMajor)?;
        let output = rusparse::host::csr_gather(pattern, dense)?;
        Ok(HostTensor::from_data(TensorData::new(output, [pattern.nnz()])))
    }

    fn csr_scatter_add(matrix: &Self::CsrHandle, values: HostTensor) -> Result<HostTensor, HostSparseError> {
        Self::csr_validate_values(matrix, &values)?;
        let pattern = matrix.as_ref().as_ref();
        let values = tensor_values(values)?;
        let matrix = CsrMatrix::new(
            pattern.rows(), pattern.columns(), pattern.row_offsets(), pattern.column_indices(), &values, pattern.index_base(),
        )?;
        let output = rusparse::host::csr_scatter_add(matrix)?;
        let output = output.as_ref();
        Ok(HostTensor::from_data(TensorData::new(output.values().to_vec(), [output.rows(), output.columns()])))
    }

    fn csr_to_dense(matrix: &Self::CsrHandle, values: HostTensor) -> Result<HostTensor, HostSparseError> {
        Self::csr_validate_values(matrix, &values)?;
        let pattern = matrix.as_ref().as_ref();
        let values = tensor_values(values)?;
        let matrix = CsrMatrix::new(
            pattern.rows(), pattern.columns(), pattern.row_offsets(), pattern.column_indices(),
            &values, pattern.index_base(),
        )?;
        let output = rusparse::conversion::csr_to_dense(matrix, DenseOrder::RowMajor)?;
        let output = output.as_ref();
        Ok(HostTensor::from_data(TensorData::new(
            output.values().to_vec(), [output.rows(), output.columns()],
        )))
    }

    fn csr_to_dense_backward(matrix: &Self::CsrHandle, grad: HostTensor) -> Result<HostTensor, HostSparseError> {
        let pattern = matrix.as_ref().as_ref();
        Self::csr_validate_operand(matrix, &grad, &HostDevice, &[pattern.rows(), pattern.columns()])?;
        let values = tensor_values(grad)?;
        let grad = DenseMatrix::new(&values, pattern.rows(), pattern.columns(), DenseOrder::RowMajor)?;
        let output = rusparse::host::csr_to_dense_backward(pattern, grad)?;
        Ok(HostTensor::from_data(TensorData::new(output, [pattern.nnz()])))
    }

    fn csr_matmul(
        matrix: &Self::CsrHandle,
        values: HostTensor,
        rhs: HostTensor,
        transpose: bool,
    ) -> Result<HostTensor, HostSparseError> {
        Self::csr_validate_values(matrix, &values)?;
        let pattern = matrix.as_ref().as_ref();
        let inner = if transpose { pattern.rows() } else { pattern.columns() };
        let columns = rhs.shape().get(1).copied().unwrap_or(0);
        Self::csr_validate_operand(matrix, &rhs, &HostDevice, &[inner, columns])?;
        let values = tensor_values(values)?;
        let rhs_values = tensor_values(rhs)?;
        let matrix = CsrMatrix::new(
            pattern.rows(), pattern.columns(), pattern.row_offsets(), pattern.column_indices(),
            &values, pattern.index_base(),
        )?;
        let rhs = DenseMatrix::new(&rhs_values, inner, columns, DenseOrder::RowMajor)?;
        let output = rusparse::host::csr_matmul(matrix, rhs, transpose)?;
        let output = output.as_ref();
        Ok(HostTensor::from_data(TensorData::new(
            output.values().to_vec(), [output.rows(), output.columns()],
        )))
    }

    fn csr_sampled_matmul(
        matrix: &Self::CsrHandle,
        lhs: HostTensor,
        rhs: HostTensor,
    ) -> Result<HostTensor, HostSparseError> {
        let pattern = matrix.as_ref().as_ref();
        let inner = lhs.shape().get(1).copied().unwrap_or(0);
        Self::csr_validate_operand(matrix, &lhs, &HostDevice, &[pattern.rows(), inner])?;
        Self::csr_validate_operand(matrix, &rhs, &HostDevice, &[inner, pattern.columns()])?;
        let lhs_values = tensor_values(lhs)?;
        let rhs_values = tensor_values(rhs)?;
        let lhs = DenseMatrix::new(&lhs_values, pattern.rows(), inner, DenseOrder::RowMajor)?;
        let rhs = DenseMatrix::new(&rhs_values, inner, pattern.columns(), DenseOrder::RowMajor)?;
        let output = rusparse::host::csr_sampled_matmul(pattern, lhs, rhs)?;
        Ok(HostTensor::from_data(TensorData::new(output.as_ref().values().to_vec(), [pattern.nnz()])))
    }
}

fn tensor_values(tensor: HostTensor) -> Result<Vec<f32>, HostSparseError> {
    Ok(tensor.to_contiguous().into_data().into_vec::<f32>()?)
}
