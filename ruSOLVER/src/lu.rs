// SPDX-License-Identifier: Apache-2.0
use crate::{
    Matrix, MatrixView, SolverError, Tolerance
};
use crate::numerics::{
    checked, sum_iter
};
/// Reusable square LU with partial ROW pivoting. Convention: `P * A = L * U`.
/// `pivots[k]` is the row interchanged with row k during elimination. The factors
/// own a copy; the caller's input is never modified. Not a blocked BLAS kernel.
#[derive(Clone, Debug)]
pub struct Lu {
    packed: Matrix, pivots: Vec<usize>, sign: i32
}
impl Lu {
    pub fn factor(a: MatrixView<'_>, tolerance: Tolerance) -> Result<Self, SolverError> {
        let n=a.square()?;
        let cutoff=tolerance.threshold(a.max_abs())?;
        let mut packed=a.to_owned()?;
        let mut pivots=Vec::new();
        pivots.try_reserve_exact(n).map_err(|_| SolverError::Allocation)?;
        let mut sign=1;
        for k in 0..n {
            let mut pivot=k;
            for i in k+1..n {
                if packed.data[i*n+k].abs()>packed.data[pivot*n+k].abs() {
                    pivot=i;
                }
            }
            let value=packed.data[pivot*n+k];
            if value.abs()<=cutoff {
                return Err(SolverError::Singular {
                    index: k, pivot: value, threshold: cutoff
                });
            }
            pivots.push(pivot);
            if pivot!=k {
                for j in 0..n {
                    packed.data.swap(k*n+j, pivot*n+j);
                }
                sign=-sign;
            }
            for i in k+1..n {
                let ratio=checked(packed.data[i*n+k]/packed.data[k*n+k], "LU multiplier")?;
                packed.data[i*n+k]=ratio;
                for j in k+1..n {
                    packed.data[i*n+j]=checked((-ratio).mul_add(packed.data[k*n+j], packed.data[i*n+j]), "LU update")?;
                }
            }
        }
        Ok(Self {
            packed, pivots, sign
        })
    }
    pub fn order(&self) -> usize {
        self.packed.rows
    }
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }
    pub fn packed(&self) -> &Matrix {
        &self.packed
    }
    pub fn lower(&self) -> Result<Matrix, SolverError> {
        let n=self.order();
        let mut l=Matrix::identity(n)?;
        for i in 0..n {
            for j in 0..i {
                l.data[i*n+j]=self.packed.data[i*n+j];
            }
        }
        Ok(l)
    }
    pub fn upper(&self) -> Result<Matrix, SolverError> {
        let n=self.order();
        let mut u=Matrix::zeros(n, n)?;
        for i in 0..n {
            for j in i..n {
                u.data[i*n+j]=self.packed.data[i*n+j];
            }
        }
        Ok(u)
    }
    /// Solve all RHS columns, reusing the factorization. RHS is not modified.
    pub fn solve(&self, b: MatrixView<'_>) -> Result<Matrix, SolverError> {
        let n=self.order();
        let m=b.columns();
        if b.rows()!=n {
            return Err(SolverError::Shape("LU RHS row count"));
        }
        let mut x=b.to_owned()?;
        for (k, &p) in self.pivots.iter().enumerate() {
            if k!=p {
                for j in 0..m {
                    x.data.swap(k*m+j, p*m+j);
                }
            }
        }
        for i in 0..n {
            for c in 0..m {
                let sum=sum_iter((0..i).map(|j| self.packed.data[i*n+j]*x.data[j*m+c]))?;
                x.data[i*m+c]=checked(x.data[i*m+c]-sum, "LU forward solve")?;
            }
        }
        for i in (0..n).rev() {
            for c in 0..m {
                let sum=sum_iter((i+1..n).map(|j| self.packed.data[i*n+j]*x.data[j*m+c]))?;
                x.data[i*m+c]=checked((x.data[i*m+c]-sum)/self.packed.data[i*n+i], "LU back solve")?;
            }
        }
        Ok(x)
    }
    /// Solve `A^T X = B` without refactorizing A.
    pub fn solve_transpose(&self, b: MatrixView<'_>) -> Result<Matrix, SolverError> {
        let n=self.order();
        let m=b.columns();
        if b.rows()!=n {
            return Err(SolverError::Shape("transposed LU RHS row count"));
        }
        let mut x=b.to_owned()?;
        for i in 0..n {
            for c in 0..m {
                let sum=sum_iter((0..i).map(|j| self.packed.data[j*n+i]*x.data[j*m+c]))?;
                x.data[i*m+c]=checked((x.data[i*m+c]-sum)/self.packed.data[i*n+i], "U transpose solve")?;
            }
        }
        for i in (0..n).rev() {
            for c in 0..m {
                let sum=sum_iter((i+1..n).map(|j| self.packed.data[j*n+i]*x.data[j*m+c]))?;
                x.data[i*m+c]=checked(x.data[i*m+c]-sum, "L transpose solve")?;
            }
        }
        for k in (0..n).rev() {
            let p=self.pivots[k];
            if k!=p {
                for j in 0..m {
                    x.data.swap(k*m+j, p*m+j);
                }
            }
        }
        Ok(x)
    }
    /// Prefer solve(B) to explicitly forming an inverse for linear systems.
    pub fn inverse(&self) -> Result<Matrix, SolverError> {
        self.solve(Matrix::identity(self.order())?.view())
    }
    /// Determinant sign and log(abs(det)), avoiding a potentially overflowing product.
    pub fn slogdet(&self) -> Result<(i32, f64), SolverError> {
        let n=self.order();
        let mut sign=self.sign;
        for i in 0..n {
            if self.packed.data[i*n+i]<0.0 {
                sign=-sign;
            }
        }
        let log=sum_iter((0..n).map(|i| self.packed.data[i*n+i].abs().ln()))?;
        Ok((sign, log))
    }
}
