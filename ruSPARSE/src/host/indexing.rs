use super::*;

pub fn csr_gather(pattern: CsrMatrix<'_>, dense: DenseMatrix<'_>) -> Result<Vec<f32>, SparseError> {
    if dense.rows() != pattern.rows() || dense.columns() != pattern.columns() {
        return Err(SparseError::DimensionMismatch("CSR gather shape mismatch"));
    }
    let mut output = vec![0.0; pattern.nnz()];
    let base = pattern.index_base().value();
    for row in 0..pattern.rows() {
        for entry in (pattern.row_offsets()[row] - base) as usize..(pattern.row_offsets()[row + 1] - base) as usize {
            output[entry] = dense_at(dense, row, (pattern.column_indices()[entry] - base) as usize);
        }
    }
    Ok(output)
}

pub fn csr_scatter_add(matrix: CsrMatrix<'_>) -> Result<DenseMatrixOwned, SparseError> {
    let length = matrix.rows().checked_mul(matrix.columns())
        .ok_or(SparseError::SizeOverflow("CSR scatter output"))?;
    let mut output = vec![0.0; length];
    let base = matrix.index_base().value();
    for row in 0..matrix.rows() {
        for entry in (matrix.row_offsets()[row] - base) as usize..(matrix.row_offsets()[row + 1] - base) as usize {
            let column = (matrix.column_indices()[entry] - base) as usize;
            output[row * matrix.columns() + column] += matrix.values()[entry];
        }
    }
    DenseMatrixOwned::new(output, matrix.rows(), matrix.columns(), DenseOrder::RowMajor)
}
