use super::*;
use crate::Operation;

pub fn sampled_csrgemm(
    pattern: CsrMatrix<'_>, operation_a: Operation, operation_b: Operation,
    a: CsrMatrix<'_>, b: CsrMatrix<'_>,
) -> Result<Vec<f32>, SparseError> {
    let transposed_a = super::binary::transpose(a, operation_a)?;
    let transposed_b = super::binary::transpose(b, operation_b)?;
    let a = transposed_a.as_ref().map_or(a, CsrMatrixOwned::as_ref);
    let b = transposed_b.as_ref().map_or(b, CsrMatrixOwned::as_ref);
    if a.columns() != b.rows() || pattern.rows() != a.rows() || pattern.columns() != b.columns() {
        return Err(SparseError::DimensionMismatch("sampled sparse product dimensions differ from pattern"));
    }
    let base = pattern.index_base().value();
    let a_base = a.index_base().value();
    let b_base = b.index_base().value();
    let mut values = vec![0.0f32; pattern.nnz()];
    for row in 0..pattern.rows() {
        for entry in (pattern.row_offsets()[row] - base) as usize..(pattern.row_offsets()[row + 1] - base) as usize {
            let column = pattern.column_indices()[entry] - base;
            let mut sum = 0.0f32;
            for left in (a.row_offsets()[row] - a_base) as usize..(a.row_offsets()[row + 1] - a_base) as usize {
                let inner = (a.column_indices()[left] - a_base) as usize;
                for right in (b.row_offsets()[inner] - b_base) as usize..(b.row_offsets()[inner + 1] - b_base) as usize {
                    if b.column_indices()[right] - b_base == column {
                        sum = a.values()[left].mul_add(b.values()[right], sum);
                    }
                }
            }
            values[entry] = sum;
        }
    }
    Ok(values)
}
