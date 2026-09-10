// SPDX-License-Identifier: Apache-2.0
//! Borrow existing ruSPARSE storage; no second owning sparse-format library.
use crate::{
    LinearOperator, SolverError, JacobiPreconditioner
};
use crate::numerics::{
    finite, sum_iter, zeros
};
use rusparse::{
    CsrMatrix, IndexBase
};
/// FP32 CSR coefficients with FP64 host vectors/accumulation. Input coefficients
/// are widened explicitly, not claimed to have originally contained FP64 precision.
/// Both ruSPARSE index bases and duplicate entries are supported by summation.
#[derive(Clone, Copy, Debug)]
pub struct CsrF32Operator<'a>{
    matrix: CsrMatrix<'a>
}
impl<'a>CsrF32Operator<'a>{
    pub fn new(matrix: CsrMatrix<'a>)->Result<Self, SolverError>{
        if matrix.rows()==0 || matrix.rows()!=matrix.columns(){
            return Err(SolverError::Shape("CSR operator must be nonempty and square"));
        }
        for (index, &x) in matrix.values().iter().enumerate(){
            if !x.is_finite(){
                return Err(SolverError::NonFinite{
                    index
                });
            }
        }
        Ok(Self{
            matrix
        })
    }
    fn base(&self)->usize{
        match self.matrix.index_base(){
            IndexBase::Zero=>0, IndexBase::One=>1
        }
    }
    pub fn jacobi(&self)->Result<JacobiPreconditioner, SolverError>{
        let n=self.dimension();
        let base=self.base();
        let mut d=zeros(n)?;
        for row in 0..n{
            let start=self.matrix.row_offsets()[row] as usize-base;
            let end=self.matrix.row_offsets()[row+1] as usize-base;
            d[row]=sum_iter((start..end).filter(|&j|self.matrix.column_indices()[j] as usize-base==row)
            .map(|j|f64::from(self.matrix.values()[j])))?;
        }
        JacobiPreconditioner::new(&d)
    }
}
impl LinearOperator for CsrF32Operator<'_>{
    fn dimension(&self)->usize{
        self.matrix.rows()
    }
    fn apply(&self, x: &[f64], out: &mut[f64])->Result<(), SolverError>{
        if x.len()!=self.dimension()||out.len()!=x.len(){
            return Err(SolverError::Shape("CSR vector length"));
        }
        finite(x)?;
        let base=self.base();
        for (row, item) in out.iter_mut().enumerate(){
            let start=self.matrix.row_offsets()[row] as usize-base;
            let end=self.matrix.row_offsets()[row+1] as usize-base;
            *item=sum_iter((start..end).map(|j|f64::from(self.matrix.values()[j])*x[self.matrix.column_indices()[j] as usize-base]))?;
        }
        Ok(())
    }
}
#[cfg(test)]mod tests{
    use super::*;
    use crate::{
        CgOptions, conjugate_gradient
    };
    #[test]
    fn bases_and_duplicate_diagonals(){
        for base in [IndexBase::Zero, IndexBase::One]{
            let shift=if base==IndexBase::One{
                1
            } else{
                0
            };
            let offsets: Vec<_>=[0, 3, 5].iter().map(|x|x+shift).collect();
            let indices: Vec<_>=[0, 0, 1, 0, 1].iter().map(|x|x+shift).collect();
            let values=[2.0, 2.0, 1.0, 1.0, 3.0];
            let csr=CsrMatrix::new(2, 2, &offsets, &indices, &values, base).unwrap();
            let a=CsrF32Operator::new(csr).unwrap();
            let m=a.jacobi().unwrap();
            let result=conjugate_gradient(&a, &[6.0, 7.0], None, &m, CgOptions::default()).unwrap();
            assert!(result.converged());
            assert!((result.solution[0]-1.0).abs()<1e-10);
            assert!((result.solution[1]-2.0).abs()<1e-10);
        }
    }
}
