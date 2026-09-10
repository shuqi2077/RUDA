//! Device-resident FP32 CSR operations on the shared Ruda tensor/runtime boundary.

use crate::{CsrMatrix, CsrMatrixOwned, DenseOrder, IndexBase, Operation, SparseError, dimension};
use ruda_core::{
    device::Device,
    tensor::{DType, data::TensorData},
};
use ruda_kernel::{
    dsl::prelude::*,
    tensor::{
        RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous,
        transfer::from_data,
    },
};

mod kernel;
mod sddmm;
mod spvv;
mod sparse_binary;
mod sampled_sparse;
mod indexing;
pub use indexing::{csr_gather, csr_scatter_add};
mod transpose;
mod conversion;
mod transfer;

pub use sddmm::sddmm;
pub use spvv::{SparseVectorTensor, spvv};
pub use sparse_binary::{csrgeam, csrgemm};
pub use sampled_sparse::sampled_csrgemm;
pub use conversion::csr_to_dense;
pub use conversion::csr_to_dense_backward;

/// Validated CSR buffers uploaded once and shared by sparse operations.
#[derive(Clone, Debug)]
pub struct CsrTensor<R: Runtime> {
    rows: usize,
    columns: usize,
    nnz: usize,
    base: IndexBase,
    offsets: RudaTensor<R>,
    indices: RudaTensor<R>,
    values: RudaTensor<R>,
    host_offsets: std::sync::Arc<[u32]>,
    host_indices: std::sync::Arc<[u32]>,
}

impl<R: Runtime> CsrTensor<R> {
    /// Upload a validated matrix, applying the requested sparse transpose before upload.
    pub fn from_csr(
        matrix: CsrMatrix<'_>,
        operation: Operation,
        device: &R::Device,
    ) -> Result<Self, SparseError> {
        let transposed = match operation {
            Operation::None => None,
            Operation::Transpose | Operation::ConjugateTranspose => Some(matrix.transpose()?),
        };
        let matrix = transposed.as_ref().map_or(matrix, CsrMatrixOwned::as_ref);
        Ok(Self {
            rows: matrix.rows(),
            columns: matrix.columns(),
            nnz: matrix.nnz(),
            base: matrix.index_base(),
            host_offsets: matrix.row_offsets().into(),
            host_indices: matrix.column_indices().into(),
            offsets: from_data(
                TensorData::new(matrix.row_offsets().to_vec(), [matrix.rows() + 1]),
                device,
            ),
            indices: from_data(
                TensorData::new(matrix.column_indices().to_vec(), [matrix.nnz()]),
                device,
            ),
            values: from_data(
                TensorData::new(matrix.values().to_vec(), [matrix.nnz()]),
                device,
            ),
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn nnz(&self) -> usize {
        self.nnz
    }
    pub fn index_base(&self) -> IndexBase {
        self.base
    }

    pub fn row_offsets(&self) -> RudaTensor<R> {
        self.offsets.clone()
    }

    pub fn column_indices(&self) -> RudaTensor<R> {
        self.indices.clone()
    }

    pub fn values(&self) -> RudaTensor<R> {
        self.values.clone()
    }

    /// Replace numeric values without changing the validated sparsity pattern.
    pub fn with_values(&self, values: RudaTensor<R>) -> Result<Self, SparseError> {
        self.validate_dense(&values, &[self.nnz])?;
        let mut output = self.clone();
        output.values = into_contiguous(values);
        Ok(output)
    }

    fn pattern(&self) -> crate::symbolic::CsrPattern<'_> {
        crate::symbolic::CsrPattern {
            rows: self.rows, index_base: self.base,
            row_offsets: &self.host_offsets, column_indices: &self.host_indices,
        }
    }

    fn from_pattern(&self, rows: usize, columns: usize, row_offsets: Vec<u32>, column_indices: Vec<u32>) -> Result<Self, SparseError> {
        self.base.value().checked_add(dimension(columns, "CSR columns")?)
            .ok_or(SparseError::SizeOverflow("CSR column index range"))?;
        let nnz = column_indices.len();
        let host_offsets = row_offsets.as_slice().into();
        let host_indices = column_indices.as_slice().into();
        Ok(Self {
            rows, columns, nnz, base: self.base, host_offsets, host_indices,
            offsets: from_data(TensorData::new(row_offsets, [rows + 1]), &self.values.device),
            indices: from_data(TensorData::new(column_indices, [nnz]), &self.values.device),
            values: empty_device_contiguous_dtype(self.values.client.clone(), self.values.device.clone(), [nnz].into(), DType::F32),
        })
    }

    fn validate_dense(&self, tensor: &RudaTensor<R>, shape: &[usize]) -> Result<(), SparseError> {
        if tensor.meta.shape()[..] != *shape {
            return Err(SparseError::DimensionMismatch(
                "sparse dense operand shape mismatch",
            ));
        }
        if tensor.dtype != DType::F32
            || tensor.qparams.is_some()
            || tensor.device.to_id() != self.values.device.to_id()
        {
            return Err(SparseError::Device(
                "sparse operands must be same-device unquantized F32 tensors",
            ));
        }
        Ok(())
    }

    fn grid(&self, elements: usize) -> Result<RudaCount, SparseError> {
        let elements = dimension(elements, "sparse output")?;
        let hardware = &self.values.client.properties().hardware;
        if hardware.plane_size_min != 32
            || hardware.plane_size_max != 32
            || hardware.max_ruda_dim.0 < 128
        {
            return Err(SparseError::Device(
                "row-split sparse kernels require 32-lane planes and 128-thread blocks",
            ));
        }
        let lanes = elements
            .checked_mul(32)
            .ok_or(SparseError::SizeOverflow("sparse launch lanes"))?;
        Ok(RudaCount::Static(lanes.div_ceil(128), 1, 1))
    }
}

/// `alpha * A * x + beta * y`, with FP32 lane-wise FMA and a 32-lane row reduction.
pub fn csrmv<R: Runtime>(
    matrix: &CsrTensor<R>,
    alpha: f32,
    x: RudaTensor<R>,
    beta: f32,
    y: RudaTensor<R>,
) -> Result<RudaTensor<R>, SparseError> {
    matrix.validate_dense(&x, &[matrix.columns])?;
    matrix.validate_dense(&y, &[matrix.rows])?;
    let grid = matrix.grid(matrix.rows)?;
    let output = empty_device_contiguous_dtype(
        matrix.values.client.clone(),
        matrix.values.device.clone(),
        [matrix.rows].into(),
        DType::F32,
    );
    if matrix.rows == 0 {
        return Ok(output);
    }
    kernel::csrmv::launch::<R>(
        &matrix.values.client,
        grid,
        RudaDim::new_1d(128),
        matrix.offsets.clone().into_array_arg(),
        matrix.indices.clone().into_array_arg(),
        matrix.values.clone().into_array_arg(),
        into_contiguous(x).into_array_arg(),
        into_contiguous(y).into_array_arg(),
        output.clone().into_array_arg(),
        matrix.rows as u32,
        matrix.base.value(),
        alpha,
        beta,
        include_str!("kernel.rs").to_owned(),
    );
    Ok(output)
}

/// `alpha * A * op(B) + beta * C`; the returned tensor preserves the requested dense order.
pub fn csrmm<R: Runtime>(
    matrix: &CsrTensor<R>,
    operation_b: Operation,
    alpha: f32,
    mut b: RudaTensor<R>,
    beta: f32,
    c: Option<RudaTensor<R>>,
    output_order: DenseOrder,
) -> Result<RudaTensor<R>, SparseError> {
    if b.meta.rank() != 2 {
        return Err(SparseError::DimensionMismatch("SpMM B must have two axes"));
    }
    if operation_b != Operation::None {
        b.meta.swap(0, 1);
    }
    let columns = b.meta.shape()[1];
    dimension(columns, "SpMM columns")?;
    matrix.validate_dense(&b, &[matrix.columns, columns])?;
    if let Some(c) = &c {
        matrix.validate_dense(c, &[matrix.rows, columns])?;
    }
    let elements = matrix
        .rows
        .checked_mul(columns)
        .ok_or(SparseError::SizeOverflow("SpMM output"))?;
    let grid = matrix.grid(elements)?;
    let (shape, row_stride, column_stride) = match output_order {
        DenseOrder::RowMajor => (
            [matrix.rows, columns],
            dimension(columns, "SpMM columns")?,
            1,
        ),
        DenseOrder::ColumnMajor => (
            [columns, matrix.rows],
            1,
            dimension(matrix.rows, "SpMM rows")?,
        ),
    };
    let mut output = empty_device_contiguous_dtype(
        matrix.values.client.clone(),
        matrix.values.device.clone(),
        shape.into(),
        DType::F32,
    );
    if output_order == DenseOrder::ColumnMajor {
        output.meta.swap(0, 1);
    }
    if elements == 0 {
        return Ok(output);
    }
    let c = c.unwrap_or_else(|| {
        ruda_kernel::tensor::initialization::zeros(
            matrix.values.device.clone(),
            [matrix.rows, columns].into(),
            DType::F32,
        )
    });
    kernel::csrmm::launch::<R>(
        &matrix.values.client,
        grid,
        RudaDim::new_1d(128),
        matrix.offsets.clone().into_array_arg(),
        matrix.indices.clone().into_array_arg(),
        matrix.values.clone().into_array_arg(),
        into_contiguous(b).into_array_arg(),
        into_contiguous(c).into_array_arg(),
        output.clone().into_array_arg(),
        matrix.rows as u32,
        columns as u32,
        matrix.base.value(),
        row_stride,
        column_stride,
        alpha,
        beta,
        include_str!("kernel.rs").to_owned(),
    );
    Ok(output)
}
