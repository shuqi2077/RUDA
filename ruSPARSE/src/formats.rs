use crate::{CsrMatrixOwned, IndexBase, SparseError};

#[derive(Debug, Clone, Copy)]
pub struct CooMatrix<'a> {
    rows: usize,
    columns: usize,
    row_indices: &'a [u32],
    column_indices: &'a [u32],
    values: &'a [f32],
    index_base: IndexBase,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CooMatrixOwned {
    rows: usize,
    columns: usize,
    row_indices: Vec<u32>,
    column_indices: Vec<u32>,
    values: Vec<f32>,
    index_base: IndexBase,
}

impl CooMatrixOwned {
    pub fn new(
        rows: usize,
        columns: usize,
        row_indices: Vec<u32>,
        column_indices: Vec<u32>,
        values: Vec<f32>,
        index_base: IndexBase,
    ) -> Result<Self, SparseError> {
        CooMatrix::new(
            rows,
            columns,
            &row_indices,
            &column_indices,
            &values,
            index_base,
        )?;
        Ok(Self {
            rows,
            columns,
            row_indices,
            column_indices,
            values,
            index_base,
        })
    }

    pub fn as_ref(&self) -> CooMatrix<'_> {
        CooMatrix {
            rows: self.rows,
            columns: self.columns,
            row_indices: &self.row_indices,
            column_indices: &self.column_indices,
            values: &self.values,
            index_base: self.index_base,
        }
    }
}

impl<'a> CooMatrix<'a> {
    pub fn new(
        rows: usize,
        columns: usize,
        row_indices: &'a [u32],
        column_indices: &'a [u32],
        values: &'a [f32],
        index_base: IndexBase,
    ) -> Result<Self, SparseError> {
        validate_dimension(rows, "COO rows")?;
        validate_dimension(columns, "COO columns")?;
        check_length("COO row indices", values.len(), row_indices.len())?;
        check_length("COO column indices", values.len(), column_indices.len())?;
        let base = index_base.value();
        let row_limit = index_limit(base, rows, "COO row range")?;
        let column_limit = index_limit(base, columns, "COO column range")?;
        if row_indices
            .iter()
            .any(|&row| row < base || row >= row_limit)
        {
            return Err(SparseError::InvalidSparseIndex("COO row"));
        }
        if column_indices
            .iter()
            .any(|&column| column < base || column >= column_limit)
        {
            return Err(SparseError::InvalidSparseIndex("COO column"));
        }
        Ok(Self {
            rows,
            columns,
            row_indices,
            column_indices,
            values,
            index_base,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub const fn nnz(&self) -> usize {
        self.values.len()
    }

    pub const fn row_indices(&self) -> &'a [u32] {
        self.row_indices
    }

    pub const fn column_indices(&self) -> &'a [u32] {
        self.column_indices
    }

    pub const fn values(&self) -> &'a [f32] {
        self.values
    }

    pub const fn index_base(&self) -> IndexBase {
        self.index_base
    }

    /// Convert COO to row-sorted CSR. Entries in the same row retain their
    /// original relative order, including duplicate coordinates.
    pub fn to_csr(self) -> Result<CsrMatrixOwned, SparseError> {
        let base = self.index_base.value();
        let mut row_offsets = vec![0_u32; self.rows + 1];
        for &row in self.row_indices {
            let row = (row - base) as usize;
            row_offsets[row] = row_offsets[row]
                .checked_add(1)
                .ok_or(SparseError::SizeOverflow("COO row counts"))?;
        }
        prefix_offsets(&mut row_offsets, base, "COO row offsets")?;
        let mut column_indices = vec![0_u32; self.nnz()];
        let mut values = vec![0.0_f32; self.nnz()];
        for entry in (0..self.nnz()).rev() {
            let row = (self.row_indices[entry] - base) as usize;
            row_offsets[row] -= 1;
            let destination = (row_offsets[row] - base) as usize;
            column_indices[destination] = self.column_indices[entry];
            values[destination] = self.values[entry];
        }
        CsrMatrixOwned::new(
            self.rows,
            self.columns,
            row_offsets,
            column_indices,
            values,
            self.index_base,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CscMatrix<'a> {
    rows: usize,
    columns: usize,
    column_offsets: &'a [u32],
    row_indices: &'a [u32],
    values: &'a [f32],
    index_base: IndexBase,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CscMatrixOwned {
    rows: usize,
    columns: usize,
    column_offsets: Vec<u32>,
    row_indices: Vec<u32>,
    values: Vec<f32>,
    index_base: IndexBase,
}

impl CscMatrixOwned {
    pub fn new(
        rows: usize,
        columns: usize,
        column_offsets: Vec<u32>,
        row_indices: Vec<u32>,
        values: Vec<f32>,
        index_base: IndexBase,
    ) -> Result<Self, SparseError> {
        CscMatrix::new(
            rows,
            columns,
            &column_offsets,
            &row_indices,
            &values,
            index_base,
        )?;
        Ok(Self {
            rows,
            columns,
            column_offsets,
            row_indices,
            values,
            index_base,
        })
    }

    pub fn as_ref(&self) -> CscMatrix<'_> {
        CscMatrix {
            rows: self.rows,
            columns: self.columns,
            column_offsets: &self.column_offsets,
            row_indices: &self.row_indices,
            values: &self.values,
            index_base: self.index_base,
        }
    }
}

impl<'a> CscMatrix<'a> {
    pub fn new(
        rows: usize,
        columns: usize,
        column_offsets: &'a [u32],
        row_indices: &'a [u32],
        values: &'a [f32],
        index_base: IndexBase,
    ) -> Result<Self, SparseError> {
        validate_dimension(rows, "CSC rows")?;
        validate_dimension(columns, "CSC columns")?;
        let expected_offsets = columns
            .checked_add(1)
            .ok_or(SparseError::SizeOverflow("CSC column offsets"))?;
        check_length("CSC column offsets", expected_offsets, column_offsets.len())?;
        check_length("CSC row indices", values.len(), row_indices.len())?;
        validate_offsets("CSC column", column_offsets, values.len(), index_base)?;
        let base = index_base.value();
        let row_limit = index_limit(base, rows, "CSC row range")?;
        if row_indices
            .iter()
            .any(|&row| row < base || row >= row_limit)
        {
            return Err(SparseError::InvalidSparseIndex("CSC row"));
        }
        Ok(Self {
            rows,
            columns,
            column_offsets,
            row_indices,
            values,
            index_base,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub const fn nnz(&self) -> usize {
        self.values.len()
    }

    pub const fn column_offsets(&self) -> &'a [u32] {
        self.column_offsets
    }

    pub const fn row_indices(&self) -> &'a [u32] {
        self.row_indices
    }

    pub const fn values(&self) -> &'a [f32] {
        self.values
    }

    pub const fn index_base(&self) -> IndexBase {
        self.index_base
    }

    pub fn to_csr(self) -> Result<CsrMatrixOwned, SparseError> {
        let base = self.index_base.value();
        let mut row_offsets = vec![0_u32; self.rows + 1];
        for &row in self.row_indices {
            let row = (row - base) as usize;
            row_offsets[row + 1] = row_offsets[row + 1]
                .checked_add(1)
                .ok_or(SparseError::SizeOverflow("CSC to CSR row counts"))?;
        }
        prefix_offsets(&mut row_offsets, base, "CSC to CSR row offsets")?;
        let mut positions = row_offsets[..self.rows]
            .iter()
            .map(|offset| offset - base)
            .collect::<Vec<_>>();
        let mut column_indices = vec![0_u32; self.nnz()];
        let mut values = vec![0.0_f32; self.nnz()];
        for column in 0..self.columns {
            let start = (self.column_offsets[column] - base) as usize;
            let end = (self.column_offsets[column + 1] - base) as usize;
            for entry in start..end {
                let row = (self.row_indices[entry] - base) as usize;
                let destination = positions[row] as usize;
                column_indices[destination] = validate_dimension(column, "CSC column")? + base;
                values[destination] = self.values[entry];
                positions[row] += 1;
            }
        }
        CsrMatrixOwned::new(
            self.rows,
            self.columns,
            row_offsets,
            column_indices,
            values,
            self.index_base,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockDirection {
    RowMajor,
    ColumnMajor,
}

#[derive(Debug, Clone, Copy)]
pub struct BsrMatrix<'a> {
    block_rows: usize,
    block_columns: usize,
    block_dimension: usize,
    row_offsets: &'a [u32],
    column_indices: &'a [u32],
    values: &'a [f32],
    direction: BlockDirection,
    index_base: IndexBase,
}

impl<'a> BsrMatrix<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        block_rows: usize,
        block_columns: usize,
        block_dimension: usize,
        row_offsets: &'a [u32],
        column_indices: &'a [u32],
        values: &'a [f32],
        direction: BlockDirection,
        index_base: IndexBase,
    ) -> Result<Self, SparseError> {
        validate_dimension(block_rows, "BSR block rows")?;
        validate_dimension(block_columns, "BSR block columns")?;
        if block_dimension == 0 {
            return Err(SparseError::InvalidBlockDimension);
        }
        validate_dimension(block_dimension, "BSR block dimension")?;
        let expected_offsets = block_rows
            .checked_add(1)
            .ok_or(SparseError::SizeOverflow("BSR row offsets"))?;
        check_length("BSR row offsets", expected_offsets, row_offsets.len())?;
        validate_offsets("BSR row", row_offsets, column_indices.len(), index_base)?;
        let block_elements = block_dimension
            .checked_mul(block_dimension)
            .ok_or(SparseError::SizeOverflow("BSR block"))?;
        let expected_values = column_indices
            .len()
            .checked_mul(block_elements)
            .ok_or(SparseError::SizeOverflow("BSR values"))?;
        check_length("BSR values", expected_values, values.len())?;
        let base = index_base.value();
        let column_limit = index_limit(base, block_columns, "BSR block-column range")?;
        if column_indices
            .iter()
            .any(|&column| column < base || column >= column_limit)
        {
            return Err(SparseError::InvalidSparseIndex("BSR block column"));
        }
        block_rows
            .checked_mul(block_dimension)
            .ok_or(SparseError::SizeOverflow("BSR rows"))?;
        block_columns
            .checked_mul(block_dimension)
            .ok_or(SparseError::SizeOverflow("BSR columns"))?;
        Ok(Self {
            block_rows,
            block_columns,
            block_dimension,
            row_offsets,
            column_indices,
            values,
            direction,
            index_base,
        })
    }

    pub const fn block_rows(&self) -> usize {
        self.block_rows
    }

    pub const fn block_columns(&self) -> usize {
        self.block_columns
    }

    pub const fn block_dimension(&self) -> usize {
        self.block_dimension
    }

    pub const fn nnzb(&self) -> usize {
        self.column_indices.len()
    }

    pub const fn direction(&self) -> BlockDirection {
        self.direction
    }

    pub const fn index_base(&self) -> IndexBase {
        self.index_base
    }

    pub fn rows(&self) -> usize {
        self.block_rows * self.block_dimension
    }

    pub fn columns(&self) -> usize {
        self.block_columns * self.block_dimension
    }

    pub fn to_csr(self) -> Result<CsrMatrixOwned, SparseError> {
        let base = self.index_base.value();
        let rows = self.rows();
        let columns = self.columns();
        let entries_per_block_row = self.block_dimension;
        let nnz = self
            .nnzb()
            .checked_mul(self.block_dimension)
            .and_then(|value| value.checked_mul(self.block_dimension))
            .ok_or(SparseError::SizeOverflow("BSR to CSR nnz"))?;
        let mut row_offsets = vec![base; rows + 1];
        let mut column_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        for block_row in 0..self.block_rows {
            let block_start = (self.row_offsets[block_row] - base) as usize;
            let block_end = (self.row_offsets[block_row + 1] - base) as usize;
            for row_in_block in 0..entries_per_block_row {
                for block_entry in block_start..block_end {
                    let block_column = (self.column_indices[block_entry] - base) as usize;
                    for column_in_block in 0..self.block_dimension {
                        let value_offset = match self.direction {
                            BlockDirection::RowMajor => {
                                row_in_block * self.block_dimension + column_in_block
                            }
                            BlockDirection::ColumnMajor => {
                                column_in_block * self.block_dimension + row_in_block
                            }
                        };
                        column_indices.push(
                            validate_dimension(
                                block_column * self.block_dimension + column_in_block,
                                "BSR to CSR column",
                            )? + base,
                        );
                        values.push(
                            self.values[block_entry * self.block_dimension * self.block_dimension
                                + value_offset],
                        );
                    }
                }
                let row = block_row * self.block_dimension + row_in_block;
                row_offsets[row + 1] = validate_dimension(column_indices.len(), "BSR to CSR nnz")?
                    .checked_add(base)
                    .ok_or(SparseError::SizeOverflow("BSR to CSR row offsets"))?;
            }
        }
        CsrMatrixOwned::new(
            rows,
            columns,
            row_offsets,
            column_indices,
            values,
            self.index_base,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EllMatrix<'a> {
    rows: usize,
    columns: usize,
    width: usize,
    column_indices: &'a [u32],
    values: &'a [f32],
    index_base: IndexBase,
}

impl<'a> EllMatrix<'a> {
    pub fn new(
        rows: usize,
        columns: usize,
        width: usize,
        column_indices: &'a [u32],
        values: &'a [f32],
        index_base: IndexBase,
    ) -> Result<Self, SparseError> {
        validate_dimension(rows, "ELL rows")?;
        validate_dimension(columns, "ELL columns")?;
        validate_dimension(width, "ELL width")?;
        let expected = rows
            .checked_mul(width)
            .ok_or(SparseError::SizeOverflow("ELL storage"))?;
        check_length("ELL column indices", expected, column_indices.len())?;
        check_length("ELL values", expected, values.len())?;
        Ok(Self {
            rows,
            columns,
            width,
            column_indices,
            values,
            index_base,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn column_indices(&self) -> &'a [u32] {
        self.column_indices
    }

    pub const fn values(&self) -> &'a [f32] {
        self.values
    }

    pub const fn index_base(&self) -> IndexBase {
        self.index_base
    }

    pub const fn padding_index() -> u32 {
        u32::MAX
    }

    /// Convert rocSPARSE's column-major ELL storage (`slot * rows + row`)
    /// to CSR, ignoring out-of-range padding columns.
    pub fn to_csr(self) -> Result<CsrMatrixOwned, SparseError> {
        let base = self.index_base.value();
        let column_limit = index_limit(base, self.columns, "ELL column range")?;
        let mut row_offsets = Vec::with_capacity(self.rows + 1);
        let mut column_indices = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(base);
        for row in 0..self.rows {
            for slot in 0..self.width {
                let entry = slot * self.rows + row;
                let column = self.column_indices[entry];
                if column >= base && column < column_limit {
                    column_indices.push(column);
                    values.push(self.values[entry]);
                }
            }
            row_offsets.push(
                validate_dimension(column_indices.len(), "ELL to CSR nnz")?
                    .checked_add(base)
                    .ok_or(SparseError::SizeOverflow("ELL to CSR row offsets"))?,
            );
        }
        CsrMatrixOwned::new(
            self.rows,
            self.columns,
            row_offsets,
            column_indices,
            values,
            self.index_base,
        )
    }
}

fn validate_dimension(value: usize, name: &'static str) -> Result<u32, SparseError> {
    u32::try_from(value).map_err(|_| SparseError::DimensionTooLarge(name))
}

fn check_length(name: &'static str, expected: usize, actual: usize) -> Result<(), SparseError> {
    if expected != actual {
        return Err(SparseError::BufferLength {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

fn index_limit(base: u32, dimension: usize, name: &'static str) -> Result<u32, SparseError> {
    base.checked_add(validate_dimension(dimension, name)?)
        .ok_or(SparseError::SizeOverflow(name))
}

fn validate_offsets(
    format: &'static str,
    offsets: &[u32],
    entries: usize,
    index_base: IndexBase,
) -> Result<(), SparseError> {
    let base = index_base.value();
    if offsets.first().copied() != Some(base) {
        return Err(SparseError::InvalidSparseOffsets {
            format,
            message: "first offset does not equal the index base",
        });
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(SparseError::InvalidSparseOffsets {
            format,
            message: "offsets are not nondecreasing",
        });
    }
    let terminal = offsets.last().copied().unwrap_or(base);
    if terminal.checked_sub(base).map(|value| value as usize) != Some(entries) {
        return Err(SparseError::InvalidSparseOffsets {
            format,
            message: "terminal offset does not match the entry count",
        });
    }
    Ok(())
}

fn prefix_offsets(offsets: &mut [u32], base: u32, name: &'static str) -> Result<(), SparseError> {
    for index in 0..offsets.len().saturating_sub(1) {
        offsets[index + 1] = offsets[index + 1]
            .checked_add(offsets[index])
            .ok_or(SparseError::SizeOverflow(name))?;
    }
    for offset in offsets {
        *offset = offset
            .checked_add(base)
            .ok_or(SparseError::SizeOverflow(name))?;
    }
    Ok(())
}
