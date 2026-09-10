use crate::{CsrMatrix, CsrMatrixOwned, DenseMatrix, DenseMatrixOwned, DenseOrder, SparseError};

mod binary;
mod sampled_sparse;
mod indexing;
pub use indexing::{csr_gather, csr_scatter_add};
pub use binary::{csrgeam, csrgemm, csr_sum_pattern, csr_product_pattern};
pub use sampled_sparse::sampled_csrgemm;

pub fn csr_matmul(
    matrix: CsrMatrix<'_>,
    rhs: DenseMatrix<'_>,
    transpose: bool,
) -> Result<DenseMatrixOwned, SparseError> {
    let transposed;
    let matrix = if transpose {
        transposed = matrix.transpose()?;
        transposed.as_ref()
    } else {
        matrix
    };
    if matrix.columns() != rhs.rows() {
        return Err(SparseError::DimensionMismatch("CSR matmul inner dimensions differ"));
    }
    let length = matrix.rows().checked_mul(rhs.columns())
        .ok_or(SparseError::SizeOverflow("CSR matmul output"))?;
    let mut output = vec![0.0_f32; length];
    let base = matrix.index_base().value();
    for row in 0..matrix.rows() {
        let start = (matrix.row_offsets()[row] - base) as usize;
        let end = (matrix.row_offsets()[row + 1] - base) as usize;
        for entry in start..end {
            let inner = (matrix.column_indices()[entry] - base) as usize;
            let value = matrix.values()[entry];
            for column in 0..rhs.columns() {
                let destination = row * rhs.columns() + column;
                output[destination] = value.mul_add(dense_at(rhs, inner, column), output[destination]);
            }
        }
    }
    DenseMatrixOwned::new(output, matrix.rows(), rhs.columns(), DenseOrder::RowMajor)
}

pub fn csr_sampled_matmul(
    pattern: CsrMatrix<'_>,
    lhs: DenseMatrix<'_>,
    rhs: DenseMatrix<'_>,
) -> Result<CsrMatrixOwned, SparseError> {
    if lhs.rows() != pattern.rows() || rhs.columns() != pattern.columns() || lhs.columns() != rhs.rows() {
        return Err(SparseError::DimensionMismatch("sampled matmul dimensions differ from CSR pattern"));
    }
    let mut values = vec![0.0_f32; pattern.nnz()];
    let base = pattern.index_base().value();
    for row in 0..pattern.rows() {
        let start = (pattern.row_offsets()[row] - base) as usize;
        let end = (pattern.row_offsets()[row + 1] - base) as usize;
        for entry in start..end {
            let column = (pattern.column_indices()[entry] - base) as usize;
            let mut sum = 0.0_f32;
            for inner in 0..lhs.columns() {
                sum = dense_at(lhs, row, inner).mul_add(dense_at(rhs, inner, column), sum);
            }
            values[entry] = sum;
        }
    }
    CsrMatrixOwned::new(
        pattern.rows(), pattern.columns(), pattern.row_offsets().to_vec(),
        pattern.column_indices().to_vec(), values, pattern.index_base(),
    )
}

fn dense_at(matrix: DenseMatrix<'_>, row: usize, column: usize) -> f32 {
    let offset = match matrix.order() {
        DenseOrder::RowMajor => row * matrix.columns() + column,
        DenseOrder::ColumnMajor => column * matrix.rows() + row,
    };
    matrix.values()[offset]
}

pub fn csr_to_dense_backward(pattern: CsrMatrix<'_>, grad: DenseMatrix<'_>) -> Result<Vec<f32>, SparseError> {
    if grad.rows() != pattern.rows() || grad.columns() != pattern.columns() {
        return Err(SparseError::DimensionMismatch("CSR to dense gradient shape mismatch"));
    }
    let mut values = vec![0.0_f32; pattern.nnz()];
    let mut columns_seen = std::collections::BTreeSet::new();
    let base = pattern.index_base().value();
    for row in 0..pattern.rows() {
        columns_seen.clear();
        let start = (pattern.row_offsets()[row] - base) as usize;
        let end = (pattern.row_offsets()[row + 1] - base) as usize;
        for entry in (start..end).rev() {
            let column = (pattern.column_indices()[entry] - base) as usize;
            if columns_seen.insert(column) {
                values[entry] = dense_at(grad, row, column);
            }
        }
    }
    Ok(values)
}
