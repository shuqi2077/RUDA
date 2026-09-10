// SPDX-License-Identifier: Apache-2.0
use crate::{
    Matrix, MatrixView, SolverError, Tolerance
};
use crate::numerics::{
    checked, norm_iter, symmetric, zeros
};
#[derive(Clone, Copy, Debug)]
pub struct EigenOptions {
    pub tolerance: Tolerance,
    pub symmetry_tolerance: Tolerance,
    pub max_sweeps: usize,
}
impl Default for EigenOptions {
    fn default()->Self {
        Self{
            tolerance: Tolerance{
                absolute: 0.0, relative: 1e-12
            },
            symmetry_tolerance: Tolerance::default(), max_sweeps: 64
        }
    }
}
#[derive(Clone, Debug)]
pub struct SymmetricEigen {
    /// Ascending eigenvalues. Eigenvectors are corresponding columns.
    pub values: Vec<f64>, pub vectors: Matrix, pub sweeps: usize,
    /// Off-diagonal Frobenius norm in original units; NOT a rigorous error bound.
    pub off_diagonal_norm: f64,
}
/// Cyclic Jacobi rotations for a real symmetric, small/moderate dense matrix.
/// Scales A by max(abs(A)); no general-complex eigensolver or cuSOLVER dispatch.
/// A zero convergence tolerance and zero sweep budget are explicitly rejected.
pub fn symmetric_eigen(a: MatrixView<'_>, options: EigenOptions)->Result<SymmetricEigen, SolverError> {
    let n=a.square()?;
    symmetric(a, options.symmetry_tolerance)?;
    options.tolerance.validate()?;
    if options.max_sweeps==0 || (options.tolerance.absolute==0.0 && options.tolerance.relative==0.0) {
        return Err(SolverError::InvalidOption("eigen tolerance and sweep budget must be positive"));
    }
    let scale=a.max_abs();
    let mut v=Matrix::identity(n)?;
    if scale==0.0 {
        return Ok(SymmetricEigen{
            values: zeros(n)?, vectors: v, sweeps: 0, off_diagonal_norm: 0.0
        });
    }
    let mut w=a.to_owned()?;
    for x in &mut w.data {
        *x/=scale;
    }
    // The lower triangle defines the accepted symmetric matrix.
    for i in 0..n {
        for j in 0..i {
            w.data[j*n+i]=w.data[i*n+j];
        }
    }
    let norm=norm_iter(w.data.iter().copied())?;
    let cutoff=(options.tolerance.absolute/scale).max(options.tolerance.relative*norm);
    for sweep in 0..=options.max_sweeps {
        let off=norm_iter((0..n).flat_map(|i|(0..i).map(move|j|(i, j)))
        .map(|(i, j)|w.data[i*n+j]))?*2.0f64.sqrt();
        if off<=cutoff {
            let mut order: Vec<usize>=(0..n).collect();
            order.sort_by(|&i, &j|w.data[i*n+i].total_cmp(&w.data[j*n+j]));
            let mut values=zeros(n)?;
            let mut vectors=Matrix::zeros(n, n)?;
            for (column, &old) in order.iter().enumerate() {
                values[column]=checked(w.data[old*n+old]*scale, "eigenvalue rescale")?;
                for i in 0..n {
                    vectors.data[i*n+column]=v.data[i*n+old];
                }
            }
            return Ok(SymmetricEigen{
                values, vectors, sweeps: sweep,
                off_diagonal_norm: checked(off*scale, "eigen residual rescale")?
            });
        }
        if sweep==options.max_sweeps {
            return Err(SolverError::NonConvergence{
                iterations: sweep, residual: checked(off*scale, "eigen residual rescale")?
            });
        }
        for p in 0..n {
            for q in p+1..n {
                let apq=w.data[p*n+q];
                if apq==0.0 {
                    continue;
                }
                let delta=(w.data[q*n+q]-w.data[p*n+p])*0.5;
                let t=if delta==0.0 {
                    1.0
                } else {
                    apq/(delta+delta.hypot(apq).copysign(delta))
                };
                let c=1.0/(1.0+t*t).sqrt();
                let s=t*c;
                let app=w.data[p*n+p];
                let aqq=w.data[q*n+q];
                w.data[p*n+p]=checked(app-t*apq, "Jacobi diagonal")?;
                w.data[q*n+q]=checked(aqq+t*apq, "Jacobi diagonal")?;
                w.data[p*n+q]=0.0;
                w.data[q*n+p]=0.0;
                for k in 0..n {
                    if k!=p && k!=q {
                        let x=w.data[k*n+p];
                        let y=w.data[k*n+q];
                        let xp=checked(c*x-s*y, "Jacobi rotation")?;
                        let yq=checked(s*x+c*y, "Jacobi rotation")?;
                        w.data[k*n+p]=xp;
                        w.data[p*n+k]=xp;
                        w.data[k*n+q]=yq;
                        w.data[q*n+k]=yq;
                    }
                }
                for k in 0..n {
                    let x=v.data[k*n+p];
                    let y=v.data[k*n+q];
                    v.data[k*n+p]=c*x-s*y;
                    v.data[k*n+q]=s*x+c*y;
                }
            }
        }
    }
    Err(SolverError::Arithmetic("unreachable eigen iteration state"))
}
