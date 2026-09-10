// SPDX-License-Identifier: Apache-2.0
use rusolver::{
    Matrix, Lu, Cholesky, Qr, symmetric_eigen, conjugate_gradient, JacobiPreconditioner,
    relative_residual
};
fn main()->Result<(), Box<dyn std::error::Error>>{
    let a=Matrix::new(3, 3, vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0])?;
    let b=Matrix::new(3, 1, vec![6.0, 10.0, 8.0])?;
    let factor=Lu::factor(a.view(), Default::default())?;
    let x=factor.solve(b.view())?;
    println!("LU x={:?}, relative residual={:e}", x.values(), relative_residual(a.view(), x.values(), b.values())?);
    let chol=Cholesky::factor(a.view(), Default::default())?;
    println!("Cholesky x={:?}, logdet={}", chol.solve(b.view())?.values(), chol.log_determinant()?);
    let design=Matrix::new(4, 2, vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0])?;
    let observations=Matrix::new(4, 1, vec![1.0, 3.0, 5.0, 7.0])?;
    let ls=Qr::factor(design.view(), Default::default())?.least_squares(observations.view())?;
    println!("fit intercept/slope={:?}, residual={:?}", ls.solution.values(), ls.residual_norms);
    let eig=symmetric_eigen(a.view(), Default::default())?;
    println!("symmetric eigenvalues={:?}", eig.values);
    let cg=conjugate_gradient(&a, b.values(), None, &JacobiPreconditioner::from_matrix(a.view())?, Default::default())?;
    if !cg.converged(){
        return Err(format!("CG not converged: {:?}", cg.status).into());
    }
    println!("CG x={:?}, iterations={}, true residual={:e}", cg.solution, cg.iterations, cg.residual_norm);
    Ok(())
}
