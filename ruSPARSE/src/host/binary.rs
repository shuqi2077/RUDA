use super::*;
use crate::{Operation, symbolic::{symbolic_product, symbolic_sum}};

pub fn csr_sum_pattern(a: CsrMatrix<'_>, b: CsrMatrix<'_>) -> Result<(CsrMatrixOwned, Vec<u32>, Vec<u32>), SparseError> {
    if a.rows() != b.rows() || a.columns() != b.columns() {
        return Err(SparseError::DimensionMismatch("CSR sum operands must have the same shape"));
    }
    let symbolic = symbolic_sum(a.into(), b.into())?;
    let left = crate::symbolic::sum_entry_mapping(a.into(), &symbolic, a.index_base())?;
    let right = crate::symbolic::sum_entry_mapping(b.into(), &symbolic, a.index_base())?;
    let values = vec![0.0f32; symbolic.column_indices.len()];
    let matrix = CsrMatrixOwned::new(a.rows(), a.columns(), symbolic.row_offsets, symbolic.column_indices, values, a.index_base())?;
    Ok((matrix, left, right))
}

pub fn csrgeam(
    operation_a: Operation, operation_b: Operation,
    alpha: f32, a: CsrMatrix<'_>, beta: f32, b: CsrMatrix<'_>,
) -> Result<CsrMatrixOwned, SparseError> {
    let transposed_a = transpose(a, operation_a)?;
    let transposed_b = transpose(b, operation_b)?;
    let a = transposed_a.as_ref().map_or(a, CsrMatrixOwned::as_ref);
    let b = transposed_b.as_ref().map_or(b, CsrMatrixOwned::as_ref);
    if a.rows() != b.rows() || a.columns() != b.columns() {
        return Err(SparseError::DimensionMismatch("op(A) and op(B) must have the same shape"));
    }
    let symbolic = symbolic_sum(a.into(), b.into())?;
    let mut values = vec![0.0_f32; symbolic.column_indices.len()];
    let output_base = a.index_base().value();
    for row in 0..a.rows() {
        let start = (symbolic.row_offsets[row] - output_base) as usize;
        let end = (symbolic.row_offsets[row + 1] - output_base) as usize;
        let columns = &symbolic.column_indices[start..end];
        for (matrix, scale) in [(a, alpha), (b, beta)] {
            let base = matrix.index_base().value();
            for entry in (matrix.row_offsets()[row] - base) as usize..(matrix.row_offsets()[row + 1] - base) as usize {
                let column = matrix.column_indices()[entry] - base + output_base;
                let destination = start + columns.binary_search(&column)
                    .map_err(|_| SparseError::InvalidColumnIndex)?;
                values[destination] = scale.mul_add(matrix.values()[entry], values[destination]);
            }
        }
    }
    CsrMatrixOwned::new(a.rows(), a.columns(), symbolic.row_offsets, symbolic.column_indices, values, a.index_base())
}

pub fn csrgemm(
    operation_a: Operation, operation_b: Operation,
    alpha: f32, a: CsrMatrix<'_>, b: CsrMatrix<'_>,
) -> Result<CsrMatrixOwned, SparseError> {
    let transposed_a = transpose(a, operation_a)?;
    let transposed_b = transpose(b, operation_b)?;
    let a = transposed_a.as_ref().map_or(a, CsrMatrixOwned::as_ref);
    let b = transposed_b.as_ref().map_or(b, CsrMatrixOwned::as_ref);
    if a.columns() != b.rows() {
        return Err(SparseError::DimensionMismatch("op(A) columns must equal op(B) rows"));
    }
    let symbolic = symbolic_product(a.into(), b.into())?;
    let mut values = vec![0.0_f32; symbolic.column_indices.len()];
    let a_base = a.index_base().value();
    let b_base = b.index_base().value();
    for row in 0..a.rows() {
        let start = (symbolic.row_offsets[row] - a_base) as usize;
        let end = (symbolic.row_offsets[row + 1] - a_base) as usize;
        let columns = &symbolic.column_indices[start..end];
        for entry in (a.row_offsets()[row] - a_base) as usize..(a.row_offsets()[row + 1] - a_base) as usize {
            let inner = (a.column_indices()[entry] - a_base) as usize;
            for other in (b.row_offsets()[inner] - b_base) as usize..(b.row_offsets()[inner + 1] - b_base) as usize {
                let column = b.column_indices()[other] - b_base + a_base;
                let destination = start + columns.binary_search(&column)
                    .map_err(|_| SparseError::InvalidColumnIndex)?;
                values[destination] = a.values()[entry].mul_add(b.values()[other], values[destination]);
            }
        }
    }
    for value in &mut values { *value *= alpha; }
    CsrMatrixOwned::new(a.rows(), b.columns(), symbolic.row_offsets, symbolic.column_indices, values, a.index_base())
}

pub(super) fn transpose(matrix: CsrMatrix<'_>, operation: Operation) -> Result<Option<CsrMatrixOwned>, SparseError> {
    match operation {
        Operation::None => Ok(None),
        Operation::Transpose | Operation::ConjugateTranspose => matrix.transpose().map(Some),
    }
}

pub fn csr_product_pattern(a: CsrMatrix<'_>, b: CsrMatrix<'_>) -> Result<CsrMatrixOwned, SparseError> {
    if a.columns() != b.rows() {
        return Err(SparseError::DimensionMismatch("CSR product inner dimensions differ"));
    }
    let symbolic = symbolic_product(a.into(), b.into())?;
    let values = vec![0.0f32; symbolic.column_indices.len()];
    CsrMatrixOwned::new(a.rows(), b.columns(), symbolic.row_offsets, symbolic.column_indices, values, a.index_base())
}
