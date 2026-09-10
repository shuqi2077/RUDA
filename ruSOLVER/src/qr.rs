// SPDX-License-Identifier: Apache-2.0
use crate::{
    Matrix, MatrixView, SolverError, Tolerance
};
use crate::numerics::{
    checked, norm_iter, sum_iter, zeros
};
/// Economy QR with recomputed column norms and column pivoting.
/// Convention: `A[:, permutation] = Q * R`, where Q has m rows and n columns.
/// Householder reflectors are stored, not a full m*m Q. Only m >= n is supported.
#[derive(Clone, Debug)]
pub struct Qr {
    packed: Matrix, tau: Vec<f64>, permutation: Vec<usize>, rank: usize
}
#[derive(Clone, Debug)]
pub struct LeastSquares {
    pub solution: Matrix, pub residual_norms: Vec<f64>, pub rank: usize
}
impl Qr {
    pub fn factor(a: MatrixView<'_>, tolerance: Tolerance)->Result<Self, SolverError> {
        let (m, n)=(a.rows(), a.columns());
        if m<n {
            return Err(SolverError::Shape("QR requires rows >= columns"));
        }
        tolerance.validate()?;
        let mut packed=a.to_owned()?;
        let mut tau=zeros(n)?;
        let mut permutation: Vec<usize>=(0..n).collect();
        let mut scale=0.0f64;
        for j in 0..n {
            scale=scale.max(norm_iter((0..m).map(|i|a.at(i, j)))?);
        }
        let cutoff=tolerance.threshold(scale)?;
        for k in 0..n {
            let mut best=k;
            let mut best_norm=-1.0;
            for j in k..n {
                let v=norm_iter((k..m).map(|i|packed.data[i*n+j]))?;
                if v>best_norm {
                    best_norm=v;
                    best=j;
                }
            }
            if best!=k {
                for i in 0..m {
                    packed.data.swap(i*n+k, i*n+best);
                }
                permutation.swap(k, best);
            }
            if best_norm==0.0 {
                continue;
            }
            let x0=packed.data[k*n+k];
            let sign=if x0>=0.0 {
                1.0
            } else {
                -1.0
            };
            let alpha=-sign*best_norm;
            // Form x/(||x||) before subtracting sign to avoid x0-alpha overflow.
            let divisor=x0/best_norm+sign;
            tau[k]=1.0+x0.abs()/best_norm;
            for i in k+1..m {
                packed.data[i*n+k]=checked((packed.data[i*n+k]/best_norm)/divisor, "Householder vector")?;
            }
            packed.data[k*n+k]=alpha;
            for j in k+1..n {
                let w=checked(tau[k]*sum_iter(std::iter::once(packed.data[k*n+j])
                .chain((k+1..m).map(|i|packed.data[i*n+k]*packed.data[i*n+j])))?, "Householder update")?;
                packed.data[k*n+j]=checked(packed.data[k*n+j]-w, "QR leading update")?;
                for i in k+1..m {
                    packed.data[i*n+j]=checked((-packed.data[i*n+k]).mul_add(w, packed.data[i*n+j]), "QR trailing update")?;
                }
            }
        }
        // Rank is a tolerance-based diagnostic, not a singular-value estimate.
        let rank=(0..n).take_while(|&k|packed.data[k*n+k].abs()>cutoff).count();
        Ok(Self{
            packed, tau, permutation, rank
        })
    }
    pub fn rank(&self)->usize {
        self.rank
    }
    pub fn permutation(&self)->&[usize] {
        &self.permutation
    }
    pub fn r(&self)->Result<Matrix, SolverError> {
        let n=self.packed.cols;
        let mut r=Matrix::zeros(n, n)?;
        for i in 0..n {
            for j in i..n {
                r.data[i*n+j]=self.packed.data[i*n+j];
            }
        }
        Ok(r)
    }
    /// Materialize thin Q only on explicit request.
    pub fn q(&self)->Result<Matrix, SolverError> {
        let (m, n)=(self.packed.rows, self.packed.cols);
        let mut q=Matrix::zeros(m, n)?;
        for i in 0..n {
            q.data[i*n+i]=1.0;
        }
        for k in (0..n).rev() {
            self.apply_reflector(k, &mut q)?;
        }
        Ok(q)
    }
    fn apply_reflector(&self, k: usize, b: &mut Matrix)->Result<(), SolverError> {
        let (m, n, p)=(self.packed.rows, self.packed.cols, b.cols);
        if self.tau[k]==0.0 {
            return Ok(());
        }
        for j in 0..p {
            let w=checked(self.tau[k]*sum_iter(std::iter::once(b.data[k*p+j])
            .chain((k+1..m).map(|i|self.packed.data[i*n+k]*b.data[i*p+j])))?, "Q application")?;
            b.data[k*p+j]=checked(b.data[k*p+j]-w, "Q application")?;
            for i in k+1..m {
                b.data[i*p+j]=checked((-self.packed.data[i*n+k]).mul_add(w, b.data[i*p+j]), "Q application")?;
            }
        }
        Ok(())
    }
    /// Full-column-rank least squares without forming A^T A or an inverse.
    /// Rank-deficient and underdetermined minimum-norm problems are NOT approximated.
    pub fn least_squares(&self, b: MatrixView<'_>)->Result<LeastSquares, SolverError> {
        let (m, n, p)=(self.packed.rows, self.packed.cols, b.columns());
        if b.rows()!=m {
            return Err(SolverError::Shape("QR RHS row count"));
        }
        if self.rank<n {
            return Err(SolverError::RankDeficient{
                rank: self.rank, columns: n
            });
        }
        let mut y=b.to_owned()?;
        for k in 0..n {
            self.apply_reflector(k, &mut y)?;
        }
        let mut residual_norms=zeros(p)?;
        for j in 0..p {
            residual_norms[j]=norm_iter((n..m).map(|i|y.data[i*p+j]))?;
        }
        for i in (0..n).rev() {
            for c in 0..p {
                let sum=sum_iter((i+1..n).map(|j|self.packed.data[i*n+j]*y.data[j*p+c]))?;
                y.data[i*p+c]=checked((y.data[i*p+c]-sum)/self.packed.data[i*n+i], "QR triangular solve")?;
            }
        }
        let mut x=Matrix::zeros(n, p)?;
        for j in 0..n {
            for c in 0..p {
                x.data[self.permutation[j]*p+c]=y.data[j*p+c];
            }
        }
        Ok(LeastSquares{
            solution: x, residual_norms, rank: self.rank
        })
    }
}
