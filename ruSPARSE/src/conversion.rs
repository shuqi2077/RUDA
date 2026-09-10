use crate::{
    CooMatrix, CooMatrixOwned, CscMatrix, CscMatrixOwned, CsrMatrix, CsrMatrixOwned, DenseMatrix,
    DenseMatrixOwned, DenseOrder, IndexBase, SparseError, dimension,
};

pub fn dense_to_csr(
    dense: DenseMatrix<'_>,
    index_base: IndexBase,
) -> Result<CsrMatrixOwned, SparseError> {
    let base = index_base.value();
    let mut row_offsets = Vec::with_capacity(dense.rows + 1);
    let mut column_indices = Vec::new();
    let mut values = Vec::new();
    row_offsets.push(base);
    for row in 0..dense.rows {
        for column in 0..dense.columns {
            let value = dense.values[dense_index(dense, row, column)];
            if value != 0.0 {
                column_indices.push(
                    dimension(column, "dense to CSR column")?
                        .checked_add(base)
                        .ok_or(SparseError::SizeOverflow("dense to CSR column"))?,
                );
                values.push(value);
            }
        }
        row_offsets.push(encoded_count(
            values.len(),
            base,
            "dense to CSR row offsets",
        )?);
    }
    CsrMatrixOwned::new(
        dense.rows,
        dense.columns,
        row_offsets,
        column_indices,
        values,
        index_base,
    )
}

pub fn dense_to_coo(
    dense: DenseMatrix<'_>,
    index_base: IndexBase,
) -> Result<CooMatrixOwned, SparseError> {
    let base = index_base.value();
    let mut row_indices = Vec::new();
    let mut column_indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..dense.rows {
        for column in 0..dense.columns {
            let value = dense.values[dense_index(dense, row, column)];
            if value != 0.0 {
                row_indices.push(encoded_index(row, base, "dense to COO row")?);
                column_indices.push(encoded_index(column, base, "dense to COO column")?);
                values.push(value);
            }
        }
    }
    CooMatrixOwned::new(
        dense.rows,
        dense.columns,
        row_indices,
        column_indices,
        values,
        index_base,
    )
}

pub fn dense_to_csc(
    dense: DenseMatrix<'_>,
    index_base: IndexBase,
) -> Result<CscMatrixOwned, SparseError> {
    let base = index_base.value();
    let mut column_offsets = Vec::with_capacity(dense.columns + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    column_offsets.push(base);
    for column in 0..dense.columns {
        for row in 0..dense.rows {
            let value = dense.values[dense_index(dense, row, column)];
            if value != 0.0 {
                row_indices.push(encoded_index(row, base, "dense to CSC row")?);
                values.push(value);
            }
        }
        column_offsets.push(encoded_count(
            values.len(),
            base,
            "dense to CSC column offsets",
        )?);
    }
    CscMatrixOwned::new(
        dense.rows,
        dense.columns,
        column_offsets,
        row_indices,
        values,
        index_base,
    )
}

pub fn csr_to_dense(
    matrix: CsrMatrix<'_>,
    order: DenseOrder,
) -> Result<DenseMatrixOwned, SparseError> {
    let mut output = zero_dense(matrix.rows, matrix.columns, order)?;
    let base = matrix.index_base.value();
    for row in 0..matrix.rows {
        let start = (matrix.row_offsets[row] - base) as usize;
        let end = (matrix.row_offsets[row + 1] - base) as usize;
        for entry in start..end {
            let column = (matrix.column_indices[entry] - base) as usize;
            let index = physical_index(matrix.rows, matrix.columns, order, row, column);
            output[index] = matrix.values[entry];
        }
    }
    DenseMatrixOwned::new(output, matrix.rows, matrix.columns, order)
}

pub fn coo_to_dense(
    matrix: CooMatrix<'_>,
    order: DenseOrder,
) -> Result<DenseMatrixOwned, SparseError> {
    let mut output = zero_dense(matrix.rows(), matrix.columns(), order)?;
    let base = matrix.index_base().value();
    for entry in 0..matrix.nnz() {
        let row = (matrix.row_indices()[entry] - base) as usize;
        let column = (matrix.column_indices()[entry] - base) as usize;
        let index = physical_index(matrix.rows(), matrix.columns(), order, row, column);
        output[index] = matrix.values()[entry];
    }
    DenseMatrixOwned::new(output, matrix.rows(), matrix.columns(), order)
}

pub fn csc_to_dense(
    matrix: CscMatrix<'_>,
    order: DenseOrder,
) -> Result<DenseMatrixOwned, SparseError> {
    let mut output = zero_dense(matrix.rows(), matrix.columns(), order)?;
    let base = matrix.index_base().value();
    for column in 0..matrix.columns() {
        let start = (matrix.column_offsets()[column] - base) as usize;
        let end = (matrix.column_offsets()[column + 1] - base) as usize;
        for entry in start..end {
            let row = (matrix.row_indices()[entry] - base) as usize;
            let index = physical_index(matrix.rows(), matrix.columns(), order, row, column);
            output[index] = matrix.values()[entry];
        }
    }
    DenseMatrixOwned::new(output, matrix.rows(), matrix.columns(), order)
}

fn dense_index(matrix: DenseMatrix<'_>, row: usize, column: usize) -> usize {
    physical_index(matrix.rows, matrix.columns, matrix.order, row, column)
}

fn physical_index(
    rows: usize,
    columns: usize,
    order: DenseOrder,
    row: usize,
    column: usize,
) -> usize {
    match order {
        DenseOrder::RowMajor => row * columns + column,
        DenseOrder::ColumnMajor => column * rows + row,
    }
}

fn zero_dense(rows: usize, columns: usize, _order: DenseOrder) -> Result<Vec<f32>, SparseError> {
    let elements = rows
        .checked_mul(columns)
        .ok_or(SparseError::SizeOverflow("sparse to dense output"))?;
    Ok(vec![0.0; elements])
}

fn encoded_index(value: usize, base: u32, name: &'static str) -> Result<u32, SparseError> {
    dimension(value, name)?
        .checked_add(base)
        .ok_or(SparseError::SizeOverflow(name))
}

fn encoded_count(value: usize, base: u32, name: &'static str) -> Result<u32, SparseError> {
    encoded_index(value, base, name)
}
