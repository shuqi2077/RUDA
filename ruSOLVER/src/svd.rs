// SPDX-License-Identifier: Apache-2.0
//! Thin real SVD by scaled one-sided cyclic Jacobi rotations. No A^T A is formed.
use crate::{Matrix, MatrixView, SolverError, Tolerance};
use crate::numerics::{checked, norm_iter, sum_iter, zeros};

#[derive(Clone, Copy, Debug)]
pub struct SvdOptions {
    /// Maximum absolute cosine between non-negligible columns.
    pub orthogonality_tolerance: f64,
    pub max_sweeps: usize,
    /// Singular values at/below this threshold are explicitly truncated to zero.
    /// The returned factorization is then a rank-truncated approximation.
    pub rank_tolerance: Tolerance,
}
impl Default for SvdOptions {
    fn default() -> Self { Self { orthogonality_tolerance: 1e-12, max_sweeps: 100,
        rank_tolerance: Tolerance { absolute: 0.0, relative: 1e-14 } } }
}
#[derive(Clone, Debug)]
pub struct Svd {
    /// m by min(m,n), including an orthonormal completion of the null columns.
    pub u: Matrix,
    /// Nonnegative, descending, explicitly thresholded singular values.
    pub singular_values: Vec<f64>,
    /// min(m,n) by n; transpose, not conjugate transpose (real inputs only).
    pub vt: Matrix,
    pub rank: usize,
    pub sweeps: usize,
    pub max_column_correlation: f64,
}
impl Svd {
    pub fn factor(a: MatrixView<'_>, options: SvdOptions) -> Result<Self, SolverError> {
        options.rank_tolerance.validate()?;
        if !options.orthogonality_tolerance.is_finite() || options.orthogonality_tolerance <= 0.0
            || options.orthogonality_tolerance >= 1.0 || options.max_sweeps == 0 {
            return Err(SolverError::InvalidOption("SVD sweep budget/tolerance"));
        }
        if a.rows() < a.columns() {
            let tall = Self::factor(a.transpose(), options)?;
            return Ok(Self { u: tall.vt.transpose()?, vt: tall.u.transpose()?,
                singular_values: tall.singular_values, rank: tall.rank, sweeps: tall.sweeps,
                max_column_correlation: tall.max_column_correlation });
        }
        let (m,n) = (a.rows(), a.columns());
        let scale = a.max_abs();
        let mut b = a.to_owned()?;
        if scale > 0.0 { for x in &mut b.data { *x /= scale; } }
        let mut v = Matrix::identity(n)?;
        let norm = norm_iter(b.data.iter().copied())?;
        let cutoff = if scale == 0.0 { 0.0 } else {
            (options.rank_tolerance.absolute/scale).max(options.rank_tolerance.relative*norm)
        };
        let mut sweeps = 0;
        let correlation;
        loop {
            let mut largest = 0.0f64;
            for p in 0..n { for q in p+1..n {
                let np = norm_iter((0..m).map(|i| b.data[i*n+p]))?;
                let nq = norm_iter((0..m).map(|i| b.data[i*n+q]))?;
                if np <= cutoff || nq <= cutoff { continue; }
                let rho = sum_iter((0..m).map(|i| (b.data[i*n+p]/np)*(b.data[i*n+q]/nq)))?;
                largest = largest.max(rho.abs());
                if rho.abs() <= options.orthogonality_tolerance || sweeps == options.max_sweeps { continue; }
                // Scale the 2x2 Gram block locally. Unlike A^T A, this is never
                // stored as a global squared-condition-number eigenproblem.
                let s = np.max(nq);
                let alpha = (np/s)*(np/s);
                let beta = (nq/s)*(nq/s);
                let gamma = rho*(np/s)*(nq/s);
                let delta = 0.5*(beta-alpha);
                let t = if delta == 0.0 { gamma.signum() } else {
                    gamma/(delta+delta.hypot(gamma).copysign(delta))
                };
                let c = 1.0/(1.0+t*t).sqrt();
                let sn = t*c;
                for i in 0..m {
                    let (x,y)=(b.data[i*n+p],b.data[i*n+q]);
                    b.data[i*n+p]=c*x-sn*y; b.data[i*n+q]=sn*x+c*y;
                }
                for i in 0..n {
                    let (x,y)=(v.data[i*n+p],v.data[i*n+q]);
                    v.data[i*n+p]=c*x-sn*y; v.data[i*n+q]=sn*x+c*y;
                }
            }}
            if largest <= options.orthogonality_tolerance { correlation=largest; break; }
            if sweeps >= options.max_sweeps {
                return Err(SolverError::NonConvergence { iterations: sweeps, residual: largest });
            }
            sweeps += 1;
        }
        let mut norms=zeros(n)?;
        for j in 0..n { norms[j]=norm_iter((0..m).map(|i| b.data[i*n+j]))?; }
        let mut order: Vec<usize>=(0..n).collect();
        order.sort_by(|&p,&q| norms[q].total_cmp(&norms[p]));
        let mut u=Matrix::zeros(m,n)?;
        let mut vt=Matrix::zeros(n,n)?;
        let mut singular_values=zeros(n)?;
        let mut rank=0;
        for (j,&old) in order.iter().enumerate() {
            for i in 0..n { vt.data[j*n+i]=v.data[i*n+old]; }
            if norms[old] > cutoff && scale > 0.0 {
                singular_values[j]=checked(norms[old]*scale,"SVD rescale")?;
                for i in 0..m { u.data[i*n+j]=b.data[i*n+old]/norms[old]; }
                rank+=1;
            } else { complete_column(&mut u,j)?; }
        }
        Ok(Self {u,vt,singular_values,rank,sweeps,max_column_correlation:correlation})
    }
    /// Moore-Penrose minimum-norm solve using the explicitly truncated SVD.
    /// Supports over/underdetermined and rank-deficient problems, multiple RHS.
    pub fn solve(&self, b: MatrixView<'_>) -> Result<Matrix,SolverError> {
        if b.rows()!=self.u.rows { return Err(SolverError::Shape("SVD RHS rows")); }
        let (m,n,k,r)=(self.u.rows,self.vt.cols,self.singular_values.len(),b.columns());
        let mut temp=Matrix::zeros(k,r)?;
        for i in 0..k { if self.singular_values[i]>0.0 { for c in 0..r {
            temp.data[i*r+c]=checked(sum_iter((0..m).map(|j|self.u.data[j*k+i]*b.at(j,c)))?
                /self.singular_values[i],"SVD solve projection")?;
        }}}
        let mut x=Matrix::zeros(n,r)?;
        for i in 0..n { for c in 0..r {
            x.data[i*r+c]=sum_iter((0..k).map(|j|self.vt.data[j*n+i]*temp.data[j*r+c]))?;
        }}
        Ok(x)
    }
    pub fn pseudo_inverse(&self)->Result<Matrix,SolverError> {
        self.solve(Matrix::identity(self.u.rows)?.view())
    }
}
fn complete_column(q: &mut Matrix, column: usize)->Result<(),SolverError> {
    let (m,n)=(q.rows,q.cols);
    let mut best=zeros(m)?; let mut best_norm=0.0f64;
    for axis in 0..m {
        let mut z=zeros(m)?; z[axis]=1.0;
        for _ in 0..2 { for j in 0..column {
            let d=sum_iter((0..m).map(|i|q.data[i*n+j]*z[i]))?;
            for i in 0..m { z[i]-=d*q.data[i*n+j]; }
        }}
        let norm=norm_iter(z.iter().copied())?;
        if norm>best_norm {best_norm=norm; best=z;}
    }
    if best_norm<=64.0*f64::EPSILON {return Err(SolverError::Breakdown("SVD null-space completion"));}
    for i in 0..m {q.data[i*n+column]=best[i]/best_norm;}
    Ok(())
}
