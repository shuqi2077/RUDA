// SPDX-License-Identifier: Apache-2.0
//! Sparse Gaussian elimination with partial row pivoting and exact fill-in.
//! Uses ordered sparse rows throughout; NEVER silently densifies or drops small fill.
//! No symbolic reuse, supernodes, parallel factorization or fill-reducing ordering.
use crate::{Matrix,MatrixView,SolverError,Tolerance};
use crate::numerics::{checked,finite};
use std::collections::BTreeMap;
#[derive(Clone,Copy,Debug)]
pub struct SparseLuOptions {pub pivot_tolerance:Tolerance,pub max_factor_nonzeros:usize}
impl Default for SparseLuOptions{fn default()->Self{Self{pivot_tolerance:Tolerance::default(),max_factor_nonzeros:10_000_000}}}
#[derive(Clone,Debug)]
pub struct SparseLu {rows:Vec<BTreeMap<usize,f64>>,pivots:Vec<usize>,nonzeros:usize}
impl SparseLu{
    /// Borrow zero-based CSR. Unsorted/duplicate column entries are summed; explicit
    /// zeros are removed. offsets must contain n+1 entries and start at zero.
    pub fn factor_csr(n:usize,offsets:&[usize],columns:&[usize],values:&[f64],options:SparseLuOptions)->Result<Self,SolverError>{
        validate_csr(n,n,offsets,columns,values)?;options.pivot_tolerance.validate()?;
        let mut rows=Vec::new();rows.try_reserve_exact(n).map_err(|_|SolverError::Allocation)?;
        let mut nonzeros=0;let mut scale=0.0f64;
        for i in 0..n{
            let mut row=BTreeMap::new();
            for p in offsets[i]..offsets[i+1]{let j=columns[p];let v=checked(row.get(&j).copied().unwrap_or(0.0)+values[p],"CSR duplicate sum")?;
                if v==0.0{row.remove(&j);}else{row.insert(j,v);}}
            nonzeros+=row.len();check_limit(nonzeros,options.max_factor_nonzeros)?;
            for &v in row.values(){scale=scale.max(v.abs());}rows.push(row);
        }
        let threshold=options.pivot_tolerance.threshold(scale)?;let mut pivots=Vec::new();
        for k in 0..n{
            let mut p=k;let mut best=0.0f64;
            for i in k..n{let v=rows[i].get(&k).copied().unwrap_or(0.0).abs();if v>best{best=v;p=i;}}
            if best<=threshold{return Err(SolverError::Singular{index:k,pivot:best,threshold});}
            rows.swap(k,p);pivots.push(p);
            let pivot=rows[k][&k];
            let upper:Vec<(usize,f64)>=rows[k].range(k+1..).map(|(&j,&x)|(j,x)).collect();
            for i in k+1..n{
                let Some(value)=rows[i].get(&k).copied()else{continue};
                let multiplier=checked(value/pivot,"sparse LU multiplier")?;
                if multiplier==0.0{rows[i].remove(&k);nonzeros-=1;}else{rows[i].insert(k,multiplier);}
                for &(j,u)in &upper{
                    let before=rows[i].get(&j).copied();let value=checked((-multiplier).mul_add(u,before.unwrap_or(0.0)),"sparse LU fill")?;
                    if value==0.0{if rows[i].remove(&j).is_some(){nonzeros-=1;}}
                    else{if before.is_none(){check_limit(nonzeros+1,options.max_factor_nonzeros)?;nonzeros+=1;}rows[i].insert(j,value);}
                }
            }
        }Ok(Self{rows,pivots,nonzeros})
    }
    pub fn order(&self)->usize{self.rows.len()}
    pub fn factor_nonzeros(&self)->usize{self.nonzeros}
    pub fn pivots(&self)->&[usize]{&self.pivots}
    /// Exposes packed factors by sparse row: strict lower L, diagonal/upper U;
    /// unit diagonal of L is implicit. No n*n output allocation.
    pub fn packed_row(&self,i:usize)->Option<&BTreeMap<usize,f64>>{self.rows.get(i)}
    pub fn solve(&self,b:MatrixView<'_>)->Result<Matrix,SolverError>{
        let(n,r)=(self.order(),b.columns());if b.rows()!=n{return Err(SolverError::Shape("sparse LU RHS"));}
        let mut x=b.to_owned()?;
        for(k,&p)in self.pivots.iter().enumerate(){for c in 0..r{x.data.swap(k*r+c,p*r+c);}}
        for i in 0..n{for(&j,&a)in self.rows[i].range(..i){for c in 0..r{x.data[i*r+c]=checked((-a).mul_add(x.data[j*r+c],x.data[i*r+c]),"sparse forward solve")?;}}}
        for i in(0..n).rev(){for(&j,&a)in self.rows[i].range(i+1..){for c in 0..r{x.data[i*r+c]=checked((-a).mul_add(x.data[j*r+c],x.data[i*r+c]),"sparse back solve")?;}}
            let d=self.rows[i][&i];for c in 0..r{x.data[i*r+c]=checked(x.data[i*r+c]/d,"sparse diagonal solve")?;}}
        Ok(x)
    }
    pub fn solve_transpose(&self,b:MatrixView<'_>)->Result<Matrix,SolverError>{
        let(n,r)=(self.order(),b.columns());if b.rows()!=n{return Err(SolverError::Shape("sparse transpose RHS"));}let mut x=b.to_owned()?;
        // Column-oriented updates using the existing sparse rows; no transpose allocation.
        for i in 0..n{for c in 0..r{x.data[i*r+c]=checked(x.data[i*r+c]/self.rows[i][&i],"sparse U transpose")?;}
            for(&j,&a)in self.rows[i].range(i+1..){for c in 0..r{x.data[j*r+c]=checked((-a).mul_add(x.data[i*r+c],x.data[j*r+c]),"sparse U transpose update")?;}}}
        for i in(0..n).rev(){for(&j,&a)in self.rows[i].range(..i){for c in 0..r{x.data[j*r+c]=checked((-a).mul_add(x.data[i*r+c],x.data[j*r+c]),"sparse L transpose update")?;}}}
        for k in(0..n).rev(){for c in 0..r{x.data.swap(k*r+c,self.pivots[k]*r+c);}}Ok(x)
    }
    #[cfg(feature="sparse")]
    pub fn from_rusparse(a:&rusparse::CsrMatrix<'_>,options:SparseLuOptions)->Result<Self,SolverError>{
        let base=if a.index_base()==rusparse::IndexBase::One{1}else{0};
        let offsets:Vec<usize>=a.row_offsets().iter().map(|&x|x as usize-base).collect();
        let columns:Vec<usize>=a.column_indices().iter().map(|&x|x as usize-base).collect();
        let values:Vec<f64>=a.values().iter().map(|&x|f64::from(x)).collect();
        if a.rows()!=a.columns(){return Err(SolverError::Shape("square ruSPARSE input required"));}
        Self::factor_csr(a.rows(),&offsets,&columns,&values,options)
    }
}
fn check_limit(required:usize,limit:usize)->Result<(),SolverError>{if required>limit{Err(SolverError::WorkspaceLimit{required,limit})}else{Ok(())}}
pub(crate)fn validate_csr(rows:usize,cols:usize,offsets:&[usize],columns:&[usize],values:&[f64])->Result<(),SolverError>{
    if cols==0||rows.checked_add(1)!=Some(offsets.len())||offsets.first()!=Some(&0)
        ||offsets.last()!=Some(&values.len())||columns.len()!=values.len(){return Err(SolverError::Shape("CSR dimensions/offsets"));}
    if offsets.windows(2).any(|w|w[0]>w[1])||columns.iter().any(|&j|j>=cols){return Err(SolverError::Shape("CSR invalid index"));}
    finite(values)
}
