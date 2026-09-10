//! Rust sparse formats, conversions and device execution.

use std::error::Error;
use std::fmt::{Display, Formatter};


pub mod conversion;
pub mod host;
mod formats;
#[cfg(feature = "serde")]
mod serde;
mod symbolic;
pub use formats::{
    BlockDirection, BsrMatrix, CooMatrix, CooMatrixOwned, CscMatrix, CscMatrixOwned, EllMatrix,
};




#[cfg(feature = "tensor")]
pub mod tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexBase {
    Zero,
    One,
}

impl IndexBase {
    const fn value(self) -> u32 {
        match self {
            Self::Zero => 0,
            Self::One => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    None,
    Transpose,
    ConjugateTranspose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenseOrder {
    RowMajor,
    ColumnMajor,
}

#[derive(Debug, Clone, Copy)]
pub struct DenseMatrix<'a> {
    values: &'a [f32],
    rows: usize,
    columns: usize,
    order: DenseOrder,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrixOwned {
    values: Vec<f32>,
    rows: usize,
    columns: usize,
    order: DenseOrder,
}

impl DenseMatrixOwned {
    pub fn new(
        values: Vec<f32>,
        rows: usize,
        columns: usize,
        order: DenseOrder,
    ) -> Result<Self, SparseError> {
        DenseMatrix::new(&values, rows, columns, order)?;
        Ok(Self {
            values,
            rows,
            columns,
            order,
        })
    }

    pub fn as_ref(&self) -> DenseMatrix<'_> {
        DenseMatrix {
            values: &self.values,
            rows: self.rows,
            columns: self.columns,
            order: self.order,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SparseVector<'a> {
    indices: &'a [u32],
    values: &'a [f32],
    size: usize,
    index_base: IndexBase,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseVectorOwned {
    indices: Vec<u32>,
    values: Vec<f32>,
    size: usize,
    index_base: IndexBase,
}

impl SparseVectorOwned {
    pub fn new(
        size: usize,
        indices: Vec<u32>,
        values: Vec<f32>,
        index_base: IndexBase,
    ) -> Result<Self, SparseError> {
        SparseVector::new(size, &indices, &values, index_base)?;
        Ok(Self { indices, values, size, index_base })
    }

    pub fn as_ref(&self) -> SparseVector<'_> {
        SparseVector {
            indices: &self.indices,
            values: &self.values,
            size: self.size,
            index_base: self.index_base,
        }
    }
}

impl<'a> SparseVector<'a> {
    pub fn new(
        size: usize,
        indices: &'a [u32],
        values: &'a [f32],
        index_base: IndexBase,
    ) -> Result<Self, SparseError> {
        dimension(size, "sparse vector size")?;
        if indices.len() != values.len() {
            return Err(SparseError::BufferLength {
                name: "sparse vector indices",
                expected: values.len(),
                actual: indices.len(),
            });
        }
        let base = index_base.value();
        let limit = base
            .checked_add(dimension(size, "sparse vector index range")?)
            .ok_or(SparseError::SizeOverflow("sparse vector index range"))?;
        if indices.iter().any(|&index| index < base || index >= limit) {
            return Err(SparseError::InvalidSparseIndex("sparse vector"));
        }
        Ok(Self {
            indices,
            values,
            size,
            index_base,
        })
    }

    pub const fn size(&self) -> usize {
        self.size
    }

    pub const fn nnz(&self) -> usize {
        self.values.len()
    }

    pub const fn indices(&self) -> &'a [u32] {
        self.indices
    }

    pub const fn values(&self) -> &'a [f32] {
        self.values
    }

    pub const fn index_base(&self) -> IndexBase {
        self.index_base
    }
}

impl<'a> DenseMatrix<'a> {
    pub fn new(
        values: &'a [f32],
        rows: usize,
        columns: usize,
        order: DenseOrder,
    ) -> Result<Self, SparseError> {
        let expected = rows
            .checked_mul(columns)
            .ok_or(SparseError::SizeOverflow("dense matrix"))?;
        if values.len() != expected {
            return Err(SparseError::BufferLength {
                name: "dense matrix",
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            values,
            rows,
            columns,
            order,
        })
    }

    pub const fn values(&self) -> &'a [f32] {
        self.values
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub const fn order(&self) -> DenseOrder {
        self.order
    }

    fn physical_strides(self) -> Result<(u32, u32), SparseError> {
        let rows = dimension(self.rows, "dense rows")?;
        let columns = dimension(self.columns, "dense columns")?;
        Ok(match self.order {
            DenseOrder::RowMajor => (columns, 1),
            DenseOrder::ColumnMajor => (1, rows),
        })
    }

    fn operation_shape(self, operation: Operation) -> (usize, usize) {
        match operation {
            Operation::None => (self.rows, self.columns),
            Operation::Transpose | Operation::ConjugateTranspose => (self.columns, self.rows),
        }
    }

    fn operation_strides(self, operation: Operation) -> Result<(u32, u32), SparseError> {
        let (row, column) = self.physical_strides()?;
        Ok(match operation {
            Operation::None => (row, column),
            Operation::Transpose | Operation::ConjugateTranspose => (column, row),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CsrMatrix<'a> {
    values: &'a [f32],
    row_offsets: &'a [u32],
    column_indices: &'a [u32],
    rows: usize,
    columns: usize,
    index_base: IndexBase,
}

impl<'a> CsrMatrix<'a> {
    pub fn new(
        rows: usize,
        columns: usize,
        row_offsets: &'a [u32],
        column_indices: &'a [u32],
        values: &'a [f32],
        index_base: IndexBase,
    ) -> Result<Self, SparseError> {
        let matrix = Self {
            values,
            row_offsets,
            column_indices,
            rows,
            columns,
            index_base,
        };
        matrix.validate()?;
        Ok(matrix)
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

    pub const fn index_base(&self) -> IndexBase {
        self.index_base
    }

    pub const fn row_offsets(&self) -> &'a [u32] {
        self.row_offsets
    }

    pub const fn column_indices(&self) -> &'a [u32] {
        self.column_indices
    }

    pub const fn values(&self) -> &'a [f32] {
        self.values
    }

    fn validate(self) -> Result<(), SparseError> {
        dimension(self.rows, "CSR rows")?;
        dimension(self.columns, "CSR columns")?;
        let expected_offsets = self
            .rows
            .checked_add(1)
            .ok_or(SparseError::SizeOverflow("CSR row offsets"))?;
        if self.row_offsets.len() != expected_offsets {
            return Err(SparseError::BufferLength {
                name: "CSR row offsets",
                expected: expected_offsets,
                actual: self.row_offsets.len(),
            });
        }
        if self.column_indices.len() != self.values.len() {
            return Err(SparseError::BufferLength {
                name: "CSR column indices",
                expected: self.values.len(),
                actual: self.column_indices.len(),
            });
        }
        let base = self.index_base.value();
        if self.row_offsets.first().copied() != Some(base) {
            return Err(SparseError::InvalidRowOffsets(
                "first row offset does not equal the index base",
            ));
        }
        for offsets in self.row_offsets.windows(2) {
            if offsets[0] > offsets[1] {
                return Err(SparseError::InvalidRowOffsets(
                    "row offsets are not nondecreasing",
                ));
            }
        }
        let terminal = self.row_offsets.last().copied().unwrap_or(base);
        let encoded_nnz = terminal
            .checked_sub(base)
            .ok_or(SparseError::InvalidRowOffsets(
                "terminal row offset is below the index base",
            ))?;
        if encoded_nnz as usize != self.values.len() {
            return Err(SparseError::InvalidRowOffsets(
                "terminal row offset does not match nnz",
            ));
        }
        let column_limit = base
            .checked_add(dimension(self.columns, "CSR columns")?)
            .ok_or(SparseError::SizeOverflow("CSR column index range"))?;
        if self
            .column_indices
            .iter()
            .any(|&column| column < base || column >= column_limit)
        {
            return Err(SparseError::InvalidColumnIndex);
        }
        Ok(())
    }

    pub fn transpose(self) -> Result<CsrMatrixOwned, SparseError> {
        self.transpose_impl(false).map(|(matrix, _)| matrix)
    }

    pub fn transpose_with_permutation(self) -> Result<(CsrMatrixOwned, Vec<u32>), SparseError> {
        self.transpose_impl(true)
    }

    fn transpose_impl(self, capture_permutation: bool) -> Result<(CsrMatrixOwned, Vec<u32>), SparseError> {
        let base = self.index_base.value();
        let mut row_offsets = vec![0_u32; self.columns + 1];
        for &encoded_column in self.column_indices {
            let column = (encoded_column - base) as usize;
            row_offsets[column + 1] = row_offsets[column + 1]
                .checked_add(1)
                .ok_or(SparseError::SizeOverflow("transposed CSR row counts"))?;
        }
        for row in 0..self.columns {
            row_offsets[row + 1] = row_offsets[row + 1]
                .checked_add(row_offsets[row])
                .ok_or(SparseError::SizeOverflow("transposed CSR row offsets"))?;
        }
        let mut positions = row_offsets[..self.columns].to_vec();
        let mut column_indices = vec![0_u32; self.nnz()];
        let mut values = vec![0.0_f32; self.nnz()];
        let mut permutation = if capture_permutation { vec![0u32; self.nnz()] } else { Vec::new() };
        for row in 0..self.rows {
            let start = (self.row_offsets[row] - base) as usize;
            let end = (self.row_offsets[row + 1] - base) as usize;
            for entry in start..end {
                let column = (self.column_indices[entry] - base) as usize;
                let destination = positions[column] as usize;
                column_indices[destination] = dimension(row, "transposed CSR column")? + base;
                values[destination] = self.values[entry];
                if capture_permutation {
                    permutation[destination] = dimension(entry, "transposed CSR permutation")?;
                }
                positions[column] += 1;
            }
        }
        for offset in &mut row_offsets {
            *offset = offset
                .checked_add(base)
                .ok_or(SparseError::SizeOverflow("transposed CSR index base"))?;
        }
        let matrix = CsrMatrixOwned::new(
            self.columns,
            self.rows,
            row_offsets,
            column_indices,
            values,
            self.index_base,
        )?;
        Ok((matrix, permutation))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrixOwned {
    values: Vec<f32>,
    row_offsets: Vec<u32>,
    column_indices: Vec<u32>,
    rows: usize,
    columns: usize,
    index_base: IndexBase,
}

impl CsrMatrixOwned {
    pub fn new(
        rows: usize,
        columns: usize,
        row_offsets: Vec<u32>,
        column_indices: Vec<u32>,
        values: Vec<f32>,
        index_base: IndexBase,
    ) -> Result<Self, SparseError> {
        CsrMatrix::new(
            rows,
            columns,
            &row_offsets,
            &column_indices,
            &values,
            index_base,
        )?;
        Ok(Self {
            values,
            row_offsets,
            column_indices,
            rows,
            columns,
            index_base,
        })
    }

    pub fn as_ref(&self) -> CsrMatrix<'_> {
        CsrMatrix {
            values: &self.values,
            row_offsets: &self.row_offsets,
            column_indices: &self.column_indices,
            rows: self.rows,
            columns: self.columns,
            index_base: self.index_base,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseAlgorithm {
    RowSplitWavefront32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparsePlan {
    pub algorithm: SparseAlgorithm,
    pub output_elements: u32,
    pub block_threads: u32,
    pub grid_blocks: u32,
}

#[derive(Debug)]
pub enum SparseError {
    BufferLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    DimensionMismatch(&'static str),
    DimensionTooLarge(&'static str),
    InvalidColumnIndex,
    InvalidSparseIndex(&'static str),
    InvalidSparseOffsets {
        format: &'static str,
        message: &'static str,
    },
    InvalidBlockDimension,
    InvalidRowOffsets(&'static str),
    SizeOverflow(&'static str),
    
    #[cfg(feature = "tensor")]
    TensorExecution(ruda_core::tensor::execution::ExecutionError),
    #[cfg(feature = "tensor")]
    TensorData(ruda_core::tensor::data::DataError),
    Device(&'static str),
}

impl Display for SparseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferLength {
                name,
                expected,
                actual,
            } => write!(formatter, "{name} needs {expected} elements, got {actual}"),
            Self::DimensionMismatch(message) => formatter.write_str(message),
            Self::DimensionTooLarge(name) => write!(formatter, "{name} does not fit u32"),
            Self::InvalidColumnIndex => formatter.write_str("CSR column index is out of bounds"),
            Self::InvalidSparseIndex(name) => write!(formatter, "{name} index is out of bounds"),
            Self::InvalidSparseOffsets { format, message } => {
                write!(formatter, "invalid {format} offsets: {message}")
            }
            Self::InvalidBlockDimension => {
                formatter.write_str("BSR block dimension must be greater than zero")
            }
            Self::InvalidRowOffsets(message) => {
                write!(formatter, "invalid CSR row offsets: {message}")
            }
            Self::SizeOverflow(name) => write!(formatter, "{name} size overflows"),
            
            #[cfg(feature = "tensor")]
            Self::TensorExecution(error) => Display::fmt(error, formatter),
            #[cfg(feature = "tensor")]
            Self::TensorData(error) => Display::fmt(error, formatter),
            Self::Device(message) => formatter.write_str(message),
        }
    }
}

impl Error for SparseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            
            #[cfg(feature = "tensor")]
            Self::TensorExecution(error) => Some(error),
            #[cfg(feature = "tensor")]
            Self::TensorData(error) => Some(error),
            _ => None,
        }
    }
}



fn dimension(value: usize, name: &'static str) -> Result<u32, SparseError> {
    u32::try_from(value).map_err(|_| SparseError::DimensionTooLarge(name))
}

fn dense_strides(
    rows: usize,
    columns: usize,
    order: DenseOrder,
    name: &'static str,
) -> Result<(u32, u32), SparseError> {
    let rows = dimension(rows, name)?;
    let columns = dimension(columns, name)?;
    Ok(match order {
        DenseOrder::RowMajor => (columns, 1),
        DenseOrder::ColumnMajor => (1, rows),
    })
}
