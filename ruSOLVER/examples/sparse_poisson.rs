// SPDX-License-Identifier: Apache-2.0
use rusolver::{
    conjugate_gradient, sparse::CsrF32Operator
};
use rusparse::{
    CsrMatrix, IndexBase
};
fn main()->Result<(), Box<dyn std::error::Error>>{
    // One-dimensional Dirichlet Poisson stencil: -x[i-1]+2*x[i]-x[i+1]=b[i].
    let n=64;
    let(mut offsets, mut indices, mut values)=(vec![0u32], Vec::new(), Vec::new());
    for i in 0..n{
        if i>0{
            indices.push((i-1) as u32);
            values.push(-1.0f32);
        }
        indices.push(i as u32);
        values.push(2.0);
        if i+1<n{
            indices.push((i+1) as u32);
            values.push(-1.0);
        }
        offsets.push(values.len() as u32);
    }
    let csr=CsrMatrix::new(n, n, &offsets, &indices, &values, IndexBase::Zero)?;
    let operator=CsrF32Operator::new(csr)?;
    let preconditioner=operator.jacobi()?;
    let b=vec![1.0; n];
    let result=conjugate_gradient(&operator, &b, None, &preconditioner, Default::default())?;
    if !result.converged(){
        return Err(format!("CG failed: {:?}", result.status).into());
    }
    println!("CSR Poisson: n={n}, iterations={}, residual={:e}", result.iterations, result.residual_norm);
    for(i, &x)in result.solution.iter().enumerate(){
        let expected=((i+1)*(n-i)) as f64/2.0;
        assert!((x-expected).abs()<1e-6);
    }
    Ok(())
}
