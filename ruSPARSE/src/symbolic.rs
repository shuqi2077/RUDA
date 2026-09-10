use crate::{CsrMatrix, IndexBase, SparseError, dimension};
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
pub(crate) struct CsrPattern<'a> {
    pub rows: usize,
    pub row_offsets: &'a [u32],
    pub column_indices: &'a [u32],
    pub index_base: IndexBase,
}

impl<'a> From<CsrMatrix<'a>> for CsrPattern<'a> {
    fn from(matrix: CsrMatrix<'a>) -> Self {
        Self {
            rows: matrix.rows(),
            row_offsets: matrix.row_offsets(),
            column_indices: matrix.column_indices(),
            index_base: matrix.index_base(),
        }
    }
}

pub(crate) struct SymbolicOutput {
    pub row_offsets: Vec<u32>,
    pub column_indices: Vec<u32>,
    pub entry_rows: Vec<u32>,
}

pub(crate) fn sum_entry_mapping(
    source: CsrPattern<'_>, output: &SymbolicOutput, output_base: IndexBase,
) -> Result<Vec<u32>, SparseError> {
    let base = source.index_base.value();
    let output_base = output_base.value();
    let mut mapping = vec![0u32; source.column_indices.len()];
    for row in 0..source.rows {
        let start = (output.row_offsets[row] - output_base) as usize;
        let end = (output.row_offsets[row + 1] - output_base) as usize;
        for entry in (source.row_offsets[row] - base) as usize..(source.row_offsets[row + 1] - base) as usize {
            let column = source.column_indices[entry] - base + output_base;
            let position = output.column_indices[start..end].binary_search(&column)
                .map_err(|_| SparseError::InvalidColumnIndex)?;
            mapping[entry] = dimension(start + position, "CSR sum entry mapping")?;
        }
    }
    Ok(mapping)
}

pub(crate) fn symbolic_sum(
    a: CsrPattern<'_>,
    b: CsrPattern<'_>,
) -> Result<SymbolicOutput, SparseError> {
    let a_base = a.index_base.value();
    let b_base = b.index_base.value();
    let output_base = a_base;
    let mut row_offsets = Vec::with_capacity(a.rows + 1);
    let mut column_indices = Vec::new();
    let mut entry_rows = Vec::new();
    row_offsets.push(output_base);
    for row in 0..a.rows {
        let mut columns = BTreeSet::new();
        for entry in
            (a.row_offsets[row] - a_base) as usize..(a.row_offsets[row + 1] - a_base) as usize
        {
            columns.insert(a.column_indices[entry] - a_base + output_base);
        }
        for entry in
            (b.row_offsets[row] - b_base) as usize..(b.row_offsets[row + 1] - b_base) as usize
        {
            columns.insert(b.column_indices[entry] - b_base + output_base);
        }
        for column in columns {
            column_indices.push(column);
            entry_rows.push(dimension(row, "SpGEAM output row")?);
        }
        row_offsets.push(
            dimension(column_indices.len(), "SpGEAM output nnz")?
                .checked_add(output_base)
                .ok_or(SparseError::SizeOverflow("SpGEAM row offsets"))?,
        );
    }
    Ok(SymbolicOutput {
        row_offsets,
        column_indices,
        entry_rows,
    })
}

pub(crate) fn symbolic_product(
    a: CsrPattern<'_>,
    b: CsrPattern<'_>,
) -> Result<SymbolicOutput, SparseError> {
    let a_base = a.index_base.value();
    let b_base = b.index_base.value();
    let output_base = a_base;
    let mut row_offsets = Vec::with_capacity(a.rows + 1);
    let mut column_indices = Vec::new();
    let mut entry_rows = Vec::new();
    row_offsets.push(output_base);
    for row in 0..a.rows {
        let mut columns = BTreeSet::new();
        let a_start = (a.row_offsets[row] - a_base) as usize;
        let a_end = (a.row_offsets[row + 1] - a_base) as usize;
        for a_entry in a_start..a_end {
            let inner = (a.column_indices[a_entry] - a_base) as usize;
            let b_start = (b.row_offsets[inner] - b_base) as usize;
            let b_end = (b.row_offsets[inner + 1] - b_base) as usize;
            for b_entry in b_start..b_end {
                columns.insert(b.column_indices[b_entry] - b_base + output_base);
            }
        }
        for column in columns {
            column_indices.push(column);
            entry_rows.push(dimension(row, "SpGEMM output row")?);
        }
        row_offsets.push(
            dimension(column_indices.len(), "SpGEMM output nnz")?
                .checked_add(output_base)
                .ok_or(SparseError::SizeOverflow("SpGEMM row offsets"))?,
        );
    }
    Ok(SymbolicOutput {
        row_offsets,
        column_indices,
        entry_rows,
    })
}
