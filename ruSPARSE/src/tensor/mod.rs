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

fn sparse_candidates<R: Runtime>(matrix: &CsrTensor<R>, elements: usize) -> Result<Vec<(&'static str, u32)>, SparseError> {
    matrix.grid(elements)?;
    let hardware = &matrix.values.client.properties().hardware;
    let mut candidates = vec![("planes_4_original", 128)];
    for (name, units) in [("planes_1", 32), ("planes_2", 64), ("planes_8", 256), ("planes_16", 512)] {
        if units <= hardware.max_ruda_dim.0.min(hardware.max_units_per_ruda)
            && (elements as u64 * 32).div_ceil(units as u64) <= hardware.max_ruda_count.0 as u64 {
            candidates.push((name, units));
        }
    }
    Ok(candidates)
}

fn sparse_signature<R: Runtime>(matrix: &CsrTensor<R>, alpha: f32, beta: f32) -> String {
    use std::hash::{Hash, Hasher};
    let mut structure = std::collections::hash_map::DefaultHasher::new();
    matrix.host_offsets.hash(&mut structure);
    matrix.host_indices.hash(&mut structure);
    format!("rows={};columns={};nnz={};base={};structure={:016x};alpha={:08x};beta={:08x}",
        matrix.rows, matrix.columns, matrix.nnz, matrix.base.value(), structure.finish(), alpha.to_bits(), beta.to_bits())
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
    let candidates = sparse_candidates(matrix, matrix.rows)?;
    if matrix.rows > 0 && ruda_kernel::tensor::tuning::is_enabled() {
        let original = matrix.clone();
        let output = ruda_kernel::tensor::tuning::execute_variants(vec![matrix.offsets.clone(), matrix.indices.clone(),
            matrix.values.clone(), x.clone(), y.clone()], "sparse_csrmv", sparse_signature(matrix, alpha, beta),
            candidates, move |inputs, units| csrmv_inner(&original, alpha, inputs[3].clone(), beta, inputs[4].clone(), units)
                .map(|value| vec![value]).map_err(|error| error.to_string()))
            .map_err(|_| SparseError::Device("sparse autotune failed without replay"))?;
        if let Some(mut outputs) = output { return Ok(outputs.remove(0)); }
    }
    csrmv_inner(matrix, alpha, x, beta, y, 128)
}

fn csrmv_inner<R: Runtime>(matrix: &CsrTensor<R>, alpha: f32, x: RudaTensor<R>, beta: f32,
    y: RudaTensor<R>, units: u32) -> Result<RudaTensor<R>, SparseError> {
    let lanes = dimension(matrix.rows, "sparse output")?.checked_mul(32)
        .ok_or(SparseError::SizeOverflow("sparse launch lanes"))?;
    let grid = RudaCount::Static(lanes.div_ceil(units), 1, 1);
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
        RudaDim::new_1d(units),
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
    let candidates = sparse_candidates(matrix, elements)?;
    if elements > 0 && ruda_kernel::tensor::tuning::is_enabled() {
        let original = matrix.clone();
        let present_c = c.is_some();
        let mut inputs = vec![matrix.offsets.clone(), matrix.indices.clone(), matrix.values.clone(), b.clone()];
        if let Some(c) = &c { inputs.push(c.clone()); }
        let output = ruda_kernel::tensor::tuning::execute_variants(inputs, "sparse_csrmm",
            format!("{};operation_b={operation_b:?};output_order={output_order:?};c={present_c}", sparse_signature(matrix, alpha, beta)),
            candidates, move |inputs, units| csrmm_inner(&original, alpha, inputs[3].clone(), beta,
                if present_c { Some(inputs[4].clone()) } else { None }, output_order, units)
                .map(|value| vec![value]).map_err(|error| error.to_string()))
            .map_err(|_| SparseError::Device("sparse autotune failed without replay"))?;
        if let Some(mut outputs) = output { return Ok(outputs.remove(0)); }
    }
    csrmm_inner(matrix, alpha, b, beta, c, output_order, 128)
}

fn csrmm_inner<R: Runtime>(matrix: &CsrTensor<R>, alpha: f32, b: RudaTensor<R>, beta: f32,
    c: Option<RudaTensor<R>>, output_order: DenseOrder, units: u32) -> Result<RudaTensor<R>, SparseError> {
    let columns = b.meta.shape()[1];
    let elements = matrix.rows.checked_mul(columns).ok_or(SparseError::SizeOverflow("SpMM output"))?;
    let lanes = dimension(elements, "sparse output")?.checked_mul(32)
        .ok_or(SparseError::SizeOverflow("sparse launch lanes"))?;
    let grid = RudaCount::Static(lanes.div_ceil(units), 1, 1);
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
        RudaDim::new_1d(units),
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
