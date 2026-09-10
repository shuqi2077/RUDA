use super::*;
use crate::{BsrMatrix, CooMatrix, CscMatrix, DenseMatrix, EllMatrix};
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::tensor::initialization::zeros;
use std::collections::BTreeSet;

impl<R: Runtime> CsrTensor<R> {
    pub fn from_coo(
        matrix: CooMatrix<'_>,
        operation: Operation,
        device: &R::Device,
    ) -> Result<Self, SparseError> {
        let matrix = matrix.to_csr()?;
        Self::from_csr(matrix.as_ref(), operation, device)
    }

    pub fn from_csc(
        matrix: CscMatrix<'_>,
        operation: Operation,
        device: &R::Device,
    ) -> Result<Self, SparseError> {
        let matrix = matrix.to_csr()?;
        Self::from_csr(matrix.as_ref(), operation, device)
    }

    pub fn from_bsr(
        matrix: BsrMatrix<'_>,
        operation: Operation,
        device: &R::Device,
    ) -> Result<Self, SparseError> {
        let matrix = matrix.to_csr()?;
        Self::from_csr(matrix.as_ref(), operation, device)
    }

    pub fn from_ell(
        matrix: EllMatrix<'_>,
        operation: Operation,
        device: &R::Device,
    ) -> Result<Self, SparseError> {
        let matrix = matrix.to_csr()?;
        Self::from_csr(matrix.as_ref(), operation, device)
    }

    pub fn from_dense(
        matrix: DenseMatrix<'_>,
        index_base: IndexBase,
        operation: Operation,
        device: &R::Device,
    ) -> Result<Self, SparseError> {
        let matrix = crate::conversion::dense_to_csr(matrix, index_base)?;
        Self::from_csr(matrix.as_ref(), operation, device)
    }
}

pub fn csr_to_dense<R: Runtime>(
    matrix: &CsrTensor<R>,
    order: DenseOrder,
) -> Result<RudaTensor<R>, SparseError> {
    let elements = matrix.rows.checked_mul(matrix.columns)
        .ok_or(SparseError::SizeOverflow("CSR to dense output"))?;
    dimension(elements, "CSR to dense output")?;
    let shape = match order {
        DenseOrder::RowMajor => [matrix.rows, matrix.columns],
        DenseOrder::ColumnMajor => [matrix.columns, matrix.rows],
    };
    let mut output = ruda_kernel::tensor::reshape::reshape(
        zeros::<R>(matrix.values.device.clone(), [elements].into(), DType::F32),
        shape.into(),
    );
    if elements == 0 || matrix.nnz == 0 {
        if order == DenseOrder::ColumnMajor { output.meta.swap(0, 1); }
        return Ok(output);
    }

    let base = matrix.base.value();
    let mut source_entries = Vec::new();
    let mut destinations = Vec::new();
    let mut columns_seen = BTreeSet::new();
    for row in 0..matrix.rows {
        columns_seen.clear();
        let start = (matrix.host_offsets[row] - base) as usize;
        let end = (matrix.host_offsets[row + 1] - base) as usize;
        for entry in (start..end).rev() {
            let column = (matrix.host_indices[entry] - base) as usize;
            if columns_seen.insert(column) {
                source_entries.push(entry as u32);
                destinations.push(match order {
                    DenseOrder::RowMajor => row * matrix.columns + column,
                    DenseOrder::ColumnMajor => column * matrix.rows + row,
                } as u32);
            }
        }
    }

    let count = source_entries.len();
    let source_entries: RudaTensor<R> = from_data(
        TensorData::new(source_entries, [count]), &matrix.values.device,
    );
    let destinations: RudaTensor<R> = from_data(
        TensorData::new(destinations, [count]), &matrix.values.device,
    );
    let ruda_dim = RudaDim::new(matrix.values.client.properties(), count);
    let ruda_count = calculate_ruda_count_elemwise(&matrix.values.client, count, ruda_dim);
    scatter_values::launch::<R>(
        &matrix.values.client,
        ruda_count,
        ruda_dim,
        source_entries.into_array_arg(),
        destinations.into_array_arg(),
        matrix.values.clone().into_array_arg(),
        output.clone().into_array_arg(),
        count as u32,
        include_str!("conversion.rs").to_owned(),
    );
    if order == DenseOrder::ColumnMajor { output.meta.swap(0, 1); }
    Ok(output)
}

#[ruda(launch)]
fn scatter_values(
    source_entries: &Array<u32>,
    destinations: &Array<u32>,
    values: &Array<f32>,
    output: &mut Array<f32>,
    count: u32,
    #[comptime] _source: String,
) {
    let entry = ABSOLUTE_POS;
    if entry < count as usize {
        let destination = destinations[entry] as usize;
        output[destination] = values[source_entries[entry] as usize];
    }
}

pub fn csr_to_dense_backward<R: Runtime>(
    matrix: &CsrTensor<R>,
    grad: RudaTensor<R>,
) -> Result<RudaTensor<R>, SparseError> {
    matrix.validate_dense(&grad, &[matrix.rows, matrix.columns])?;
    let elements = matrix.rows.checked_mul(matrix.columns)
        .ok_or(SparseError::SizeOverflow("CSR to dense gradient"))?;
    dimension(elements, "CSR to dense gradient")?;
    let output = zeros::<R>(matrix.values.device.clone(), [matrix.nnz].into(), DType::F32);
    if matrix.nnz == 0 {
        return Ok(output);
    }
    let mut source_positions = Vec::new();
    let mut entries = Vec::new();
    let mut columns_seen = BTreeSet::new();
    let base = matrix.base.value();
    for row in 0..matrix.rows {
        columns_seen.clear();
        let start = (matrix.host_offsets[row] - base) as usize;
        let end = (matrix.host_offsets[row + 1] - base) as usize;
        for entry in (start..end).rev() {
            let column = (matrix.host_indices[entry] - base) as usize;
            if columns_seen.insert(column) {
                source_positions.push((row * matrix.columns + column) as u32);
                entries.push(entry as u32);
            }
        }
    }
    let count = entries.len();
    let source_positions: RudaTensor<R> = from_data(
        TensorData::new(source_positions, [count]), &matrix.values.device,
    );
    let entries: RudaTensor<R> = from_data(
        TensorData::new(entries, [count]), &matrix.values.device,
    );
    let ruda_dim = RudaDim::new(matrix.values.client.properties(), count);
    let ruda_count = calculate_ruda_count_elemwise(&matrix.values.client, count, ruda_dim);
    scatter_values::launch::<R>(
        &matrix.values.client, ruda_count, ruda_dim,
        source_positions.into_array_arg(), entries.into_array_arg(),
        into_contiguous(grad).into_array_arg(), output.clone().into_array_arg(),
        count as u32, include_str!("conversion.rs").to_owned(),
    );
    Ok(output)
}
