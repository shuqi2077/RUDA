// SPDX-License-Identifier: Apache-2.0
use crate::{
    Matrix, MatrixView, SolverError, Tolerance
};
use crate::numerics::{
    checked, sum_iter, symmetric
};
#[derive(Clone, Copy, Debug)]
pub struct CholeskyOptions {
    /// Full input symmetry is checked. Accepted roundoff differences are not
    /// averaged: the lower triangle defines the effective symmetric matrix.
    pub symmetry_tolerance: Tolerance,
    /// Reject pivots <= max(abs, rel*max_abs(A)). Default only rejects <= zero.
    pub pivot_tolerance: Tolerance,
}
impl Default for CholeskyOptions {
    fn default()->Self {
        Self {
            symmetry_tolerance: Tolerance::default(), pivot_tolerance: Tolerance::EXACT
        }
    }
}
#[derive(Clone, Debug)]
pub struct Cholesky {
    lower: Matrix
}
impl Cholesky {
    /// `A = L L^T`; non-positive-definite inputs return an error, never a hidden jitter.
    pub fn factor(a: MatrixView<'_>, options: CholeskyOptions)->Result<Self, SolverError> {
        let n=a.square()?;
        symmetric(a, options.symmetry_tolerance)?;
        let cutoff=options.pivot_tolerance.threshold(a.max_abs())?;
        let mut l=Matrix::zeros(n, n)?;
        for j in 0..n {
            let sum=sum_iter((0..j).map(|k| l.data[j*n+k]*l.data[j*n+k]))?;
            let pivot=checked(a.at(j, j)-sum, "Cholesky diagonal")?;
            if pivot<=cutoff {
                return Err(SolverError::NotPositiveDefinite{
                    index: j, pivot
                });
            }
            let diagonal=pivot.sqrt();
            l.data[j*n+j]=diagonal;
            for i in j+1..n {
                let sum=sum_iter((0..j).map(|k| l.data[i*n+k]*l.data[j*n+k]))?;
                l.data[i*n+j]=checked((a.at(i, j)-sum)/diagonal, "Cholesky column")?;
            }
        }
        Ok(Self{
            lower: l
        })
    }
    pub fn lower(&self)->&Matrix {
        &self.lower
    }
    pub fn order(&self)->usize {
        self.lower.rows
    }
    pub fn solve(&self, b: MatrixView<'_>)->Result<Matrix, SolverError> {
        let n=self.order();
        let m=b.columns();
        if b.rows()!=n {
            return Err(SolverError::Shape("Cholesky RHS row count"));
        }
        let mut x=b.to_owned()?;
        for i in 0..n {
            for c in 0..m {
                let sum=sum_iter((0..i).map(|k|self.lower.data[i*n+k]*x.data[k*m+c]))?;
                x.data[i*m+c]=checked((x.data[i*m+c]-sum)/self.lower.data[i*n+i], "Cholesky forward solve")?;
            }
        }
        for i in (0..n).rev() {
            for c in 0..m {
                let sum=sum_iter((i+1..n).map(|k|self.lower.data[k*n+i]*x.data[k*m+c]))?;
                x.data[i*m+c]=checked((x.data[i*m+c]-sum)/self.lower.data[i*n+i], "Cholesky back solve")?;
            }
        }
        Ok(x)
    }
    pub fn log_determinant(&self)->Result<f64, SolverError> {
        checked(2.0*sum_iter((0..self.order()).map(|i|self.lower.data[i*self.order()+i].ln()))?, "Cholesky log determinant")
    }
}
