// SPDX-License-Identifier: Apache-2.0
use crate::SolverError;
use crate::numerics::{
    finite, zeros
};
/// Owned finite row-major FP64 data. Empty axes are rejected in this first release.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub(crate) rows: usize, pub(crate) cols: usize, pub(crate) data: Vec<f64>
}
impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, SolverError> {
        let size = checked_shape(rows, cols)?;
        if data.len() != size {
            return Err(SolverError::Shape("rows * columns != data length"));
        }
        finite(&data)?;
        Ok(Self {
            rows, cols, data
        })
    }
    /// Explicit FP32 -> FP64 host conversion; no device transfer occurs.
    pub fn from_f32(rows: usize, cols: usize, data: &[f32]) -> Result<Self, SolverError> {
        let size = checked_shape(rows, cols)?;
        if data.len() != size {
            return Err(SolverError::Shape("FP32 input length"));
        }
        let mut out = zeros(size)?;
        for (i, &x) in data.iter().enumerate() {
            out[i] = f64::from(x);
        }
        Self::new(rows, cols, out)
    }
    pub fn zeros(rows: usize, cols: usize) -> Result<Self, SolverError> {
        let size = checked_shape(rows, cols)?;
        Ok(Self {
            rows, cols, data: zeros(size)?
        })
    }
    pub fn identity(n: usize) -> Result<Self, SolverError> {
        let mut result = Self::zeros(n, n)?;
        for i in 0..n {
            result.data[i*n+i] = 1.0;
        }
        Ok(result)
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.cols
    }
    pub fn values(&self) -> &[f64] {
        &self.data
    }
    pub fn into_values(self) -> Vec<f64> {
        self.data
    }
    pub fn view(&self) -> MatrixView<'_> {
        MatrixView {
            data: &self.data, rows: self.rows, cols: self.cols,
            offset: 0, row_stride: self.cols, col_stride: 1
        }
    }
    pub fn get(&self, row: usize, column: usize) -> Option<f64> {
        self.view().get(row, column)
    }
    pub fn transpose(&self) -> Result<Self, SolverError> {
        self.view().transpose().to_owned()
    }
}
/// Immutable matrix view. Constructor validates reachable indices and values;
/// unrelated padding elements may contain arbitrary values. Overlapping READS
/// are allowed; the view cannot be used to create mutable aliases.
#[derive(Clone, Copy, Debug)]
pub struct MatrixView<'a> {
    data: &'a [f64], rows: usize, cols: usize,
    offset: usize, row_stride: usize, col_stride: usize,
}
impl<'a> MatrixView<'a> {
    pub fn contiguous(rows: usize, cols: usize, data: &'a [f64]) -> Result<Self, SolverError> {
        if checked_shape(rows, cols)? != data.len() {
            return Err(SolverError::Shape("contiguous data length"));
        }
        Self::strided(rows, cols, data, 0, cols, 1)
    }
    pub fn strided(rows: usize, cols: usize, data: &'a [f64], offset: usize,
    row_stride: usize, col_stride: usize) -> Result<Self, SolverError> {
        checked_shape(rows, cols)?;
        if row_stride == 0 || col_stride == 0 {
            return Err(SolverError::Shape("strides must be positive"));
        }
        let last = (rows-1).checked_mul(row_stride)
        .and_then(|r| (cols-1).checked_mul(col_stride).and_then(|c| r.checked_add(c)))
        .and_then(|x| offset.checked_add(x)).ok_or(SolverError::SizeOverflow)?;
        if last >= data.len() {
            return Err(SolverError::Shape("strided view exceeds backing slice"));
        }
        let view = Self {
            data, rows, cols, offset, row_stride, col_stride
        };
        for i in 0..rows {
            for j in 0..cols {
                if !view.at(i, j).is_finite() {
                    return Err(SolverError::NonFinite {
                        index: i*cols+j
                    });
                }
            }
        }
        Ok(view)
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn columns(self) -> usize {
        self.cols
    }
    pub fn get(self, row: usize, column: usize) -> Option<f64> {
        (row < self.rows && column < self.cols).then(|| self.at(row, column))
    }
    pub fn transpose(self) -> Self {
        Self {
            rows: self.cols, cols: self.rows, row_stride: self.col_stride,
            col_stride: self.row_stride, ..self
        }
    }
    pub fn to_owned(self) -> Result<Matrix, SolverError> {
        let mut out = Matrix::zeros(self.rows, self.cols)?;
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[i*self.cols+j] = self.at(i, j);
            }
        }
        Ok(out)
    }
    pub(crate) fn at(self, row: usize, col: usize) -> f64 {
        self.data[self.offset + row*self.row_stride + col*self.col_stride]
    }
    pub(crate) fn max_abs(self) -> f64 {
        let mut max = 0.0f64;
        for i in 0..self.rows {
            for j in 0..self.cols {
                max = max.max(self.at(i, j).abs());
            }
        }
        max
    }
    pub(crate) fn square(self) -> Result<usize, SolverError> {
        if self.rows != self.cols {
            Err(SolverError::Shape("square matrix required"))
        } else {
            Ok(self.rows)
        }
    }
}
fn checked_shape(rows: usize, cols: usize) -> Result<usize, SolverError> {
    if rows == 0 || cols == 0 {
        return Err(SolverError::Shape("zero axes are not supported"));
    }
    rows.checked_mul(cols).filter(|x| *x <= isize::MAX as usize / std::mem::size_of::<f64>())
    .ok_or(SolverError::SizeOverflow)
}
