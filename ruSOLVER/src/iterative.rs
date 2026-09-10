// SPDX-License-Identifier: Apache-2.0
use crate::{
    Matrix, MatrixView, SolverError, Tolerance, l2_norm
};
use crate::numerics::{
    checked, dot, finite, sum_iter, zeros
};
/// Synchronous HOST linear map. Implementations must fully overwrite output,
/// preserve inputs and implement a fixed linear map throughout one solve.
/// Device-resident asynchronous algorithms will use a separate API, not hidden copies.
pub trait LinearOperator {
    fn dimension(&self)->usize;
    fn apply(&self, x: &[f64], output: &mut[f64])->Result<(), SolverError>;
}
pub trait Preconditioner {
    fn dimension(&self)->usize;
    fn apply(&self, r: &[f64], output: &mut[f64])->Result<(), SolverError>;
}
impl LinearOperator for Matrix {
    fn dimension(&self)->usize {
        self.rows
    }
    fn apply(&self, x: &[f64], out: &mut[f64])->Result<(), SolverError> {
        self.view().apply(x, out)
    }
}
impl LinearOperator for MatrixView<'_> {
    fn dimension(&self)->usize {
        self.rows()
    }
    fn apply(&self, x: &[f64], out: &mut[f64])->Result<(), SolverError> {
        let n=self.square()?;
        if x.len()!=n || out.len()!=n {
            return Err(SolverError::Shape("operator vector length"));
        }
        finite(x)?;
        for i in 0..n {
            out[i]=sum_iter((0..n).map(|j|self.at(i, j)*x[j]))?;
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct IdentityPreconditioner {
    n: usize
}
impl IdentityPreconditioner {
    pub fn new(n: usize)->Result<Self, SolverError> {
        if n==0 {
            Err(SolverError::Shape("empty preconditioner"))
        } else {
            Ok(Self{
                n
            })
        }
    }
}
impl Preconditioner for IdentityPreconditioner {
    fn dimension(&self)->usize {
        self.n
    }
    fn apply(&self, r: &[f64], out: &mut[f64])->Result<(), SolverError> {
        if r.len()!=self.n || out.len()!=self.n {
            return Err(SolverError::Shape("identity preconditioner"));
        }
        out.copy_from_slice(r);
        Ok(())
    }
}
/// Positive diagonal preconditioner. Reciprocal overflow is rejected at construction.
#[derive(Clone, Debug)]
pub struct JacobiPreconditioner {
    inverse: Vec<f64>
}
impl JacobiPreconditioner {
    pub fn new(diagonal: &[f64])->Result<Self, SolverError> {
        if diagonal.is_empty() {
            return Err(SolverError::Shape("empty Jacobi diagonal"));
        }
        finite(diagonal)?;
        let mut inverse=zeros(diagonal.len())?;
        for (i, &d) in diagonal.iter().enumerate() {
            if d<=0.0 {
                return Err(SolverError::NotPositiveDefinite{
                    index: i, pivot: d
                });
            }
            inverse[i]=checked(1.0/d, "Jacobi inverse diagonal")?;
        }
        Ok(Self{
            inverse
        })
    }
    pub fn from_matrix(a: MatrixView<'_>)->Result<Self, SolverError> {
        let n=a.square()?;
        let mut d=zeros(n)?;
        for i in 0..n {
            d[i]=a.at(i, i);
        }
        Self::new(&d)
    }
}
impl Preconditioner for JacobiPreconditioner {
    fn dimension(&self)->usize {
        self.inverse.len()
    }
    fn apply(&self, r: &[f64], out: &mut[f64])->Result<(), SolverError> {
        if r.len()!=self.inverse.len() || out.len()!=r.len() {
            return Err(SolverError::Shape("Jacobi vector length"));
        }
        for i in 0..r.len() {
            out[i]=checked(r[i]*self.inverse[i], "Jacobi application")?;
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct CgOptions {
    pub tolerance: Tolerance, pub max_iterations: usize,
    /// A true residual b-Ax is recomputed at this interval and before success.
    /// Replacement restarts the search direction, trading some speed for reliability.
    pub residual_recompute_interval: usize,
}
impl Default for CgOptions {
    fn default()->Self {
        Self{
            tolerance: Tolerance{
                absolute: 0.0, relative: 1e-10
            }, max_iterations: 1000, residual_recompute_interval: 32
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IterativeStatus {
    Converged, MaxIterations
}
#[derive(Clone, Debug)]
pub struct CgReport {
    pub solution: Vec<f64>, pub iterations: usize, pub residual_norm: f64,
    pub rhs_norm: f64, pub status: IterativeStatus, pub matvecs: usize
}
impl CgReport {
    pub fn converged(&self)->bool {
        self.status==IterativeStatus::Converged
    }
}
fn checked_apply(a: &impl LinearOperator, x: &[f64], out: &mut[f64])->Result<(), SolverError> {
    out.fill(f64::NAN);
    a.apply(x, out)?;
    finite(out)
}
fn checked_precondition(m: &impl Preconditioner, r: &[f64], out: &mut[f64])->Result<(), SolverError> {
    out.fill(f64::NAN);
    m.apply(r, out)?;
    finite(out)
}
fn true_residual(a: &impl LinearOperator, b: &[f64], x: &[f64], r: &mut[f64], scratch: &mut[f64])->Result<f64, SolverError> {
    checked_apply(a, x, scratch)?;
    for i in 0..r.len() {
        r[i]=checked(b[i]-scratch[i], "CG true residual")?;
    }
    l2_norm(r)
}
/// Preconditioned conjugate gradient for a fixed real symmetric POSITIVE-DEFINITE
/// operator and positive-definite preconditioner. Caller guarantees those global
/// properties; positive-curvature checks can detect some violations, not prove SPD.
/// Zero RHS and an already-correct initial guess converge in zero iterations.
/// Failure to reach tolerance returns MaxIterations, never a false success.
pub fn conjugate_gradient(a: &impl LinearOperator, b: &[f64], initial: Option<&[f64]>,
preconditioner: &impl Preconditioner, options: CgOptions)->Result<CgReport, SolverError> {
    let n=a.dimension();
    if n==0 || b.len()!=n || preconditioner.dimension()!=n {
        return Err(SolverError::Shape("CG dimension"));
    }
    options.tolerance.validate()?;
    if options.residual_recompute_interval==0 || (options.tolerance.absolute==0.0 && options.tolerance.relative==0.0) {
        return Err(SolverError::InvalidOption("CG tolerance and residual interval must be positive"));
    }
    finite(b)?;
    let mut x=zeros(n)?;
    if let Some(initial)=initial {
        if initial.len()!=n {
            return Err(SolverError::Shape("CG initial guess"));
        }
        finite(initial)?;
        x.copy_from_slice(initial);
    }
    let (mut r, mut z, mut p, mut ap)=(zeros(n)?, zeros(n)?, zeros(n)?, zeros(n)?);
    let rhs_norm=l2_norm(b)?;
    let target=options.tolerance.threshold(rhs_norm)?;
    let mut matvecs=1;
    let mut residual_norm=true_residual(a, b, &x, &mut r, &mut ap)?;
    let report=|solution, iterations, residual_norm, status, matvecs|CgReport{
        solution, iterations, residual_norm, rhs_norm, status, matvecs
    };
    if residual_norm<=target {
        return Ok(report(x, 0, residual_norm, IterativeStatus::Converged, matvecs));
    }
    if options.max_iterations==0 {
        return Ok(report(x, 0, residual_norm, IterativeStatus::MaxIterations, matvecs));
    }
    checked_precondition(preconditioner, &r, &mut z)?;
    let mut rho=dot(&r, &z)?;
    if rho<=0.0 {
        return Err(SolverError::Breakdown("nonpositive r^T M^-1 r (or numerical underflow)"));
    }
    p.copy_from_slice(&z);
    for iteration in 1..=options.max_iterations {
        checked_apply(a, &p, &mut ap)?;
        matvecs+=1;
        let curvature=dot(&p, &ap)?;
        if curvature<=0.0 {
            return Err(SolverError::Breakdown("nonpositive p^T A p; SPD required"));
        }
        let alpha=checked(rho/curvature, "CG step size")?;
        for i in 0..n {
            x[i]=checked(alpha.mul_add(p[i], x[i]), "CG iterate")?;
            r[i]=checked((-alpha).mul_add(ap[i], r[i]), "CG recursive residual")?;
        }
        residual_norm=l2_norm(&r)?;
        let replace=iteration%options.residual_recompute_interval==0 || residual_norm<=target || iteration==options.max_iterations;
        if replace {
            residual_norm=true_residual(a, b, &x, &mut r, &mut ap)?;
            matvecs+=1;
        }
        if residual_norm<=target {
            return Ok(report(x, iteration, residual_norm, IterativeStatus::Converged, matvecs));
        }
        if iteration==options.max_iterations {
            break;
        }
        checked_precondition(preconditioner, &r, &mut z)?;
        let next_rho=dot(&r, &z)?;
        if next_rho<=0.0 {
            return Err(SolverError::Breakdown("preconditioned residual underflow/nonpositive"));
        }
        if replace {
            p.copy_from_slice(&z);
        } else {
            let beta=checked(next_rho/rho, "CG direction coefficient")?;
            for i in 0..n {
                p[i]=checked(beta.mul_add(p[i], z[i]), "CG direction")?;
            }
        }
        rho=next_rho;
    }
    Ok(report(x, options.max_iterations, residual_norm, IterativeStatus::MaxIterations, matvecs))
}
