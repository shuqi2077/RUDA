// SPDX-License-Identifier: Apache-2.0
//! Real FP64 reverse-mode primitives. These differentiate the mathematical
//! operation, NOT iterations/pivot choices. No finite differences in production.
//! See ruda-autodiff's opt-in `solver-host` module for actual graph integration.
use crate::{Matrix,MatrixView,Lu,Cholesky,CholeskyOptions,Svd,SvdOptions,
    SymmetricEigen,EigenOptions,symmetric_eigen,Qr,Tolerance,SolverError};
use crate::numerics::{checked,sum_iter,finite};
pub(crate)fn mm(a:MatrixView<'_>,b:MatrixView<'_>)->Result<Matrix,SolverError>{
    if a.columns()!=b.rows(){return Err(SolverError::Shape("adjoint matmul"));}let mut c=Matrix::zeros(a.rows(),b.columns())?;
    for i in 0..a.rows(){for j in 0..b.columns(){c.data[i*b.columns()+j]=sum_iter((0..a.columns()).map(|k|a.at(i,k)*b.at(k,j)))?;}}Ok(c)
}
fn sym(a:Matrix)->Result<Matrix,SolverError>{let n=a.view().square()?;let mut x=a.clone();for i in 0..n{for j in 0..n{x.data[i*n+j]=0.5*a.data[i*n+j]+0.5*a.data[j*n+i];}}Ok(x)}
fn same(a:MatrixView<'_>,b:&Matrix)->Result<(),SolverError>{if a.rows()!=b.rows||a.columns()!=b.cols{Err(SolverError::Shape("cotangent shape"))}else{Ok(())}}
#[derive(Clone,Debug)]pub struct SolvePullback{lu:Lu,x:Matrix}
pub fn solve_with_pullback(a:MatrixView<'_>,b:MatrixView<'_>,tol:Tolerance)->Result<(Matrix,SolvePullback),SolverError>{
    let lu=Lu::factor(a,tol)?;let x=lu.solve(b)?;Ok((x.clone(),SolvePullback{lu,x}))
}
impl SolvePullback{
    /// X=A^-1 B: dB=A^-T dX; dA=-dB X^T. Multiple RHS supported.
    pub fn backward(&self,dx:MatrixView<'_>)->Result<(Matrix,Matrix),SolverError>{same(dx,&self.x)?;
        let db=self.lu.solve_transpose(dx)?;let mut da=mm(db.view(),self.x.view().transpose())?;for x in &mut da.data{*x=-*x;}Ok((da,db))}
}
#[derive(Clone,Debug)]pub struct CholeskyPullback{lower:Matrix,inverse:Matrix}
pub fn cholesky_with_pullback(a:MatrixView<'_>,options:CholeskyOptions)->Result<(Matrix,CholeskyPullback),SolverError>{
    let lower=Cholesky::factor(a,options)?.lower().clone();
    let inverse=Lu::factor(lower.view(),Tolerance::EXACT)?.inverse()?;
    Ok((lower.clone(),CholeskyPullback{lower,inverse}))
}
impl CholeskyPullback{
    /// Symmetric-input convention: returns a symmetric cotangent to the full A.
    /// Upper-triangle entries of dL are ignored because L's upper triangle is zero.
    pub fn backward(&self,dl:MatrixView<'_>)->Result<Matrix,SolverError>{
        same(dl,&self.lower)?;let n=self.lower.rows;let mut g=dl.to_owned()?;for i in 0..n{for j in i+1..n{g.data[i*n+j]=0.0;}}
        let mut p=mm(self.lower.view().transpose(),g.view())?;
        for i in 0..n{for j in 0..n{if j>i{p.data[i*n+j]=0.0;}else if i==j{p.data[i*n+j]*=0.5;}}}
        let tmp=mm(self.inverse.view().transpose(),p.view())?;sym(mm(tmp.view(),self.inverse.view())?)
    }
}
#[derive(Clone,Debug)]pub struct SvdPullback{factor:Svd,gap_tolerance:f64}
pub fn svd_with_pullback(a:MatrixView<'_>,options:SvdOptions,gap_tolerance:f64)->Result<(Svd,SvdPullback),SolverError>{
    if !gap_tolerance.is_finite()||gap_tolerance<0.0{return Err(SolverError::InvalidOption("SVD gradient gap tolerance"));}
    let factor=Svd::factor(a,options)?;Ok((factor.clone(),SvdPullback{factor,gap_tolerance}))
}
impl SvdPullback{
    /// Singular-value-only VJP. Rejects zero/repeated values rather than returning
    /// an arbitrary nonsmooth derivative. Values gradients must be finite.
    pub fn values_backward(&self,ds:&[f64])->Result<Matrix,SolverError>{
        self.validate()?;finite(ds)?;let s=&self.factor.singular_values;let k=s.len();
        if ds.len()!=k{return Err(SolverError::Shape("singular value cotangent"));}
        let mut us=self.factor.u.clone();for i in 0..us.rows{for j in 0..k{us.data[i*k+j]*=ds[j];}}mm(us.view(),self.factor.vt.view())
    }
    fn validate(&self)->Result<(),SolverError>{
        let s=&self.factor.singular_values;let scale=s[0];if self.factor.rank!=s.len()||scale==0.0{return Err(SolverError::Breakdown("SVD gradient requires full rank"));}
        for i in 0..s.len(){if s[i]/scale<=self.gap_tolerance{return Err(SolverError::Breakdown("SVD gradient near zero singular value"));}
            for j in i+1..s.len(){if ((s[i]-s[j])/scale).abs()<=self.gap_tolerance{return Err(SolverError::Breakdown("SVD gradient requires separated singular values"));}}}Ok(())
    }
    /// Full thin-SVD VJP for nonzero, distinct singular values. Singular vector
    /// signs/bases must be used consistently with this exact forward result.
    pub fn backward(&self,du:MatrixView<'_>,ds:&[f64],dvt:MatrixView<'_>)->Result<Matrix,SolverError>{
        self.validate()?;same(du,&self.factor.u)?;same(dvt,&self.factor.vt)?;finite(ds)?;
        let f=&self.factor;let k=f.singular_values.len();if ds.len()!=k{return Err(SolverError::Shape("SVD ds"));}
        let au=mm(f.u.view().transpose(),du)?;let av=mm(f.vt.view(),dvt.transpose())?;let mut middle=Matrix::zeros(k,k)?;
        for i in 0..k{middle.data[i*k+i]=ds[i];for j in 0..k{if i!=j{
            let(si,sj)=(f.singular_values[i],f.singular_values[j]);let scale=si.max(sj);let pi=si/scale;let pj=sj/scale;
            middle.data[i*k+j]=checked(((au.data[i*k+j]-au.data[j*k+i])*pj+pi*(av.data[i*k+j]-av.data[j*k+i]))
                /((pj-pi)*(pj+pi))/scale,"SVD cotangent gap")?;
        }}}
        let mut da=mm(mm(f.u.view(),middle.view())?.view(),f.vt.view())?;
        let proj_u=mm(f.u.view(),au.view())?;let mut pu=du.to_owned()?;
        for i in 0..pu.rows{for j in 0..k{pu.data[i*k+j]=(pu.data[i*k+j]-proj_u.data[i*k+j])/f.singular_values[j];}}
        let extra_u=mm(pu.view(),f.vt.view())?;
        let dv=dvt.transpose().to_owned()?;let proj_v=mm(f.vt.view().transpose(),av.view())?;let mut pv=dv;
        for i in 0..pv.rows{for j in 0..k{pv.data[i*k+j]=(pv.data[i*k+j]-proj_v.data[i*k+j])/f.singular_values[j];}}
        let extra_v=mm(f.u.view(),pv.view().transpose())?;
        for i in 0..da.data.len(){da.data[i]=checked(da.data[i]+extra_u.data[i]+extra_v.data[i],"SVD cotangent")?;}Ok(da)
    }
}
#[derive(Clone,Debug)]pub struct EigenPullback{factor:SymmetricEigen,gap_tolerance:f64}
pub fn eigen_with_pullback(a:MatrixView<'_>,options:EigenOptions,gap_tolerance:f64)->Result<(SymmetricEigen,EigenPullback),SolverError>{
    if !gap_tolerance.is_finite()||gap_tolerance<0.0{return Err(SolverError::InvalidOption("eigen gradient gap tolerance"));}
    let factor=symmetric_eigen(a,options)?;Ok((factor.clone(),EigenPullback{factor,gap_tolerance}))
}
impl EigenPullback{
    pub fn backward(&self,dl:&[f64],dv:Option<MatrixView<'_>>)->Result<Matrix,SolverError>{
        let f=&self.factor;let n=f.values.len();finite(dl)?;if dl.len()!=n{return Err(SolverError::Shape("eigenvalue cotangent"));}
        let scale=f.values.iter().fold(0.0f64,|s,&x|s.max(x.abs()));
        for i in 0..n{for j in i+1..n{if (scale==0.0||(f.values[j]/scale-f.values[i]/scale).abs()<=self.gap_tolerance){return Err(SolverError::Breakdown("eigen gradient requires distinct eigenvalues"));}}}
        let mut middle=Matrix::zeros(n,n)?;for i in 0..n{middle.data[i*n+i]=dl[i];}
        if let Some(dv)=dv{same(dv,&f.vectors)?;let vtg=mm(f.vectors.view().transpose(),dv)?;
            for i in 0..n{for j in 0..n{if i!=j{middle.data[i*n+j]=checked(0.5*(vtg.data[i*n+j]-vtg.data[j*n+i])/(f.values[j]/scale-f.values[i]/scale)/scale,"eigenvector cotangent")?;}}}}
        sym(mm(mm(f.vectors.view(),middle.view())?.view(),f.vectors.view().transpose())?)
    }
}
#[derive(Clone,Debug)]pub struct QrPullback{q:Matrix,r:Matrix,permutation:Vec<usize>}
pub fn qr_with_pullback(a:MatrixView<'_>,tol:Tolerance)->Result<(Qr,QrPullback),SolverError>{
    let qr=Qr::factor(a,tol)?;if qr.rank()!=a.columns(){return Err(SolverError::RankDeficient{rank:qr.rank(),columns:a.columns()});}
    let pb=QrPullback{q:qr.q()?,r:qr.r()?,permutation:qr.permutation().to_vec()};Ok((qr,pb))
}
impl QrPullback{
    /// Local derivative with a FIXED pivot permutation; nonsmooth pivot switches
    /// are not differentiated. Returns cotangent in the ORIGINAL column order.
    pub fn backward(&self,dq:MatrixView<'_>,dr:MatrixView<'_>)->Result<Matrix,SolverError>{
        same(dq,&self.q)?;same(dr,&self.r)?;let n=self.r.rows;
        let mut dr=dr.to_owned()?;for i in 0..n{for j in 0..i{dr.data[i*n+j]=0.0;}}
        let a=mm(self.r.view(),dr.view().transpose())?;let b=mm(dq.transpose(),self.q.view())?;
        let mut c=Matrix::zeros(n,n)?;for i in 0..n{for j in 0..=i{let x=a.data[i*n+j]-b.data[i*n+j];c.data[i*n+j]=x;c.data[j*n+i]=x;}}
        let mut rhs=mm(self.q.view(),c.view())?;for i in 0..rhs.rows{for j in 0..n{rhs.data[i*n+j]+=dq.at(i,j);}}
        // rhs R^-T = (R^-1 rhs^T)^T
        let z=Lu::factor(self.r.view(),Tolerance::EXACT)?.solve(rhs.view().transpose())?.transpose()?;
        let mut result=Matrix::zeros(z.rows,n)?;for i in 0..z.rows{for j in 0..n{result.data[i*n+self.permutation[j]]=z.data[i*n+j];}}Ok(result)
    }
}
