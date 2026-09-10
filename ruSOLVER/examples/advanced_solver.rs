// SPDX-License-Identifier: Apache-2.0
use rusolver::{Matrix,Svd};
use rusolver::complex::{Complex64 as C,ComplexMatrix,ComplexLu};
use rusolver::sparse_direct::SparseLu;
fn main()->Result<(),Box<dyn std::error::Error>>{
    let a=Matrix::new(2,3,vec![1.,0.,0.,0.,2.,0.])?;
    let svd=Svd::factor(a.view(),Default::default())?;
    let x=svd.solve(Matrix::new(2,1,vec![1.,4.])?.view())?;
    println!("minimum norm x={:?}; singular values={:?}",x.values(),svd.singular_values);
    let complex=ComplexMatrix::new(2,2,vec![C::new(2.,1.),C::ONE,C::ZERO,C::new(3.,-1.)])?;
    let rhs=ComplexMatrix::new(2,1,vec![C::ONE,C::new(2.,0.)])?;
    println!("complex LU solve={:?}",ComplexLu::factor(&complex,Default::default())?.solve(&rhs)?.values());
    let sparse=SparseLu::factor_csr(2,&[0,2,4],&[0,1,0,1],&[4.,1.,1.,3.],Default::default())?;
    println!("sparse LU solve={:?}",sparse.solve(Matrix::new(2,1,vec![6.,7.])?.view())?.values());
    let b=Matrix::new(2,1,vec![1.,2.])?;let square=Matrix::new(2,2,vec![4.,1.,1.,3.])?;
    let(x,pullback)=rusolver::adjoint::solve_with_pullback(square.view(),b.view(),Default::default())?;
    let(da,db)=pullback.backward(Matrix::new(2,1,vec![1.,1.])?.view())?;
    println!("solve={:?}, gradient A={:?}, gradient B={:?}",x.values(),da.values(),db.values());Ok(())
}
