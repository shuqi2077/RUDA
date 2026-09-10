// SPDX-License-Identifier: Apache-2.0
//! Host complex FP64 LU, Hermitian Cholesky and thin Householder QR.
//! Each component is f64 (128 storage bits per complex value); not CUDA Complex64.
use crate::{SolverError,Tolerance};
use std::ops::{Add,Sub,Mul,Div,Neg};
#[derive(Clone,Copy,Debug,Default,PartialEq)]
pub struct Complex64 {pub re:f64,pub im:f64}
impl Complex64 {
    pub const ZERO:Self=Self{re:0.0,im:0.0}; pub const ONE:Self=Self{re:1.0,im:0.0};
    pub const fn new(re:f64,im:f64)->Self{Self{re,im}}
    pub fn conj(self)->Self{Self::new(self.re,-self.im)}
    pub fn abs(self)->f64{self.re.hypot(self.im)}
    pub fn is_finite(self)->bool{self.re.is_finite()&&self.im.is_finite()}
    pub fn scale(self,s:f64)->Self{Self::new(self.re*s,self.im*s)}
}
impl Add for Complex64 {type Output=Self;fn add(self,b:Self)->Self{Self::new(self.re+b.re,self.im+b.im)}}
impl Sub for Complex64 {type Output=Self;fn sub(self,b:Self)->Self{Self::new(self.re-b.re,self.im-b.im)}}
impl Neg for Complex64 {type Output=Self;fn neg(self)->Self{Self::new(-self.re,-self.im)}}
impl Mul for Complex64 {type Output=Self;fn mul(self,b:Self)->Self{Self::new(self.re.mul_add(b.re,-self.im*b.im),self.re.mul_add(b.im,self.im*b.re))}}
impl Div for Complex64 {
    type Output=Self;
    fn div(self,b:Self)->Self {
        // Scaled denominator avoids squaring |b|. Algorithms check output finiteness.
        let s=b.re.abs().max(b.im.abs()); let br=b.re/s;let bi=b.im/s;
        let d=br*br+bi*bi;let ar=self.re/s;let ai=self.im/s;
        Self::new((ar*br+ai*bi)/d,(ai*br-ar*bi)/d)
    }
}
fn check(x:Complex64)->Result<Complex64,SolverError>{if x.is_finite(){Ok(x)}else{Err(SolverError::Arithmetic("complex intermediate"))}}
#[derive(Clone,Debug,PartialEq)]
pub struct ComplexMatrix {pub(crate) rows:usize,pub(crate) cols:usize,pub(crate) data:Vec<Complex64>}
impl ComplexMatrix {
    pub fn new(rows:usize,cols:usize,data:Vec<Complex64>)->Result<Self,SolverError>{
        if rows==0||cols==0{return Err(SolverError::Shape("empty complex matrix"));}
        let n=rows.checked_mul(cols).ok_or(SolverError::SizeOverflow)?;
        if n!=data.len(){return Err(SolverError::Shape("complex matrix length"));}
        for(i,x)in data.iter().enumerate(){if !x.is_finite(){return Err(SolverError::NonFinite{index:i});}}
        Ok(Self{rows,cols,data})
    }
    pub fn zeros(rows:usize,cols:usize)->Result<Self,SolverError>{
        let n=rows.checked_mul(cols).filter(|&n|n<=isize::MAX as usize/16).ok_or(SolverError::SizeOverflow)?;
        let mut v=Vec::new();v.try_reserve_exact(n).map_err(|_|SolverError::Allocation)?;v.resize(n,Complex64::ZERO);
        Self::new(rows,cols,v)
    }
    pub fn identity(n:usize)->Result<Self,SolverError>{let mut a=Self::zeros(n,n)?;for i in 0..n{a.data[i*n+i]=Complex64::ONE;}Ok(a)}
    pub fn rows(&self)->usize{self.rows} pub fn columns(&self)->usize{self.cols}
    pub fn values(&self)->&[Complex64]{&self.data}
    pub fn adjoint(&self)->Result<Self,SolverError>{let mut b=Self::zeros(self.cols,self.rows)?;for i in 0..self.rows{for j in 0..self.cols{b.data[j*self.rows+i]=self.data[i*self.cols+j].conj();}}Ok(b)}
    fn square(&self)->Result<usize,SolverError>{if self.rows!=self.cols{Err(SolverError::Shape("complex square matrix required"))}else{Ok(self.rows)}}
    fn max_abs(&self)->f64{self.data.iter().fold(0.0f64,|s,x|s.max(x.abs()))}
}
#[derive(Clone,Debug)]
pub struct ComplexLu {packed:ComplexMatrix,pivots:Vec<usize>}
impl ComplexLu {
    /// P A = L U; partial row pivoting, no input mutation.
    pub fn factor(a:&ComplexMatrix,tolerance:Tolerance)->Result<Self,SolverError>{
        let n=a.square()?;let cutoff=tolerance.threshold(a.max_abs())?;let mut packed=a.clone();let mut pivots=Vec::new();
        for k in 0..n{
            let mut p=k;for i in k+1..n{if packed.data[i*n+k].abs()>packed.data[p*n+k].abs(){p=i;}}
            let pivot=packed.data[p*n+k].abs();if pivot<=cutoff{return Err(SolverError::Singular{index:k,pivot,threshold:cutoff});}
            pivots.push(p);if p!=k{for j in 0..n{packed.data.swap(k*n+j,p*n+j);}}
            for i in k+1..n{let r=check(packed.data[i*n+k]/packed.data[k*n+k])?;packed.data[i*n+k]=r;
                for j in k+1..n{packed.data[i*n+j]=check(packed.data[i*n+j]-r*packed.data[k*n+j])?;}}
        }Ok(Self{packed,pivots})
    }
    pub fn packed(&self)->&ComplexMatrix{&self.packed} pub fn pivots(&self)->&[usize]{&self.pivots}
    pub fn solve(&self,b:&ComplexMatrix)->Result<ComplexMatrix,SolverError>{self.solve_impl(b,false)}
    pub fn solve_adjoint(&self,b:&ComplexMatrix)->Result<ComplexMatrix,SolverError>{self.solve_impl(b,true)}
    fn solve_impl(&self,b:&ComplexMatrix,adjoint:bool)->Result<ComplexMatrix,SolverError>{
        let n=self.packed.rows;let r=b.cols;if b.rows!=n{return Err(SolverError::Shape("complex LU RHS"));}let mut x=b.clone();
        if !adjoint{
            for(k,&p)in self.pivots.iter().enumerate(){for c in 0..r{x.data.swap(k*r+c,p*r+c);}}
            for i in 0..n{for c in 0..r{let mut s=x.data[i*r+c];for j in 0..i{s=check(s-self.packed.data[i*n+j]*x.data[j*r+c])?;}x.data[i*r+c]=s;}}
            for i in(0..n).rev(){for c in 0..r{let mut s=x.data[i*r+c];for j in i+1..n{s=check(s-self.packed.data[i*n+j]*x.data[j*r+c])?;}x.data[i*r+c]=check(s/self.packed.data[i*n+i])?;}}
        }else{
            for i in 0..n{for c in 0..r{let mut s=x.data[i*r+c];for j in 0..i{s=check(s-self.packed.data[j*n+i].conj()*x.data[j*r+c])?;}x.data[i*r+c]=check(s/self.packed.data[i*n+i].conj())?;}}
            for i in(0..n).rev(){for c in 0..r{let mut s=x.data[i*r+c];for j in i+1..n{s=check(s-self.packed.data[j*n+i].conj()*x.data[j*r+c])?;}x.data[i*r+c]=s;}}
            for k in(0..n).rev(){for c in 0..r{x.data.swap(k*r+c,self.pivots[k]*r+c);}}
        }Ok(x)
    }
}
#[derive(Clone,Debug)]pub struct ComplexCholesky{lower:ComplexMatrix}
impl ComplexCholesky{
    /// A=L L^H. Lower triangle defines the matrix after Hermitian validation.
    pub fn factor(a:&ComplexMatrix,tolerance:Tolerance)->Result<Self,SolverError>{
        let n=a.square()?;tolerance.validate()?;let cutoff=tolerance.threshold(a.max_abs())?;
        for i in 0..n{if a.data[i*n+i].im.abs()>cutoff{return Err(SolverError::NotSymmetric{row:i,column:i});}
            for j in 0..i{let x=a.data[i*n+j];let y=a.data[j*n+i].conj();let s=x.abs().max(y.abs());
                if s>0.0&&(x.scale(1.0/s)-y.scale(1.0/s)).abs()>(tolerance.absolute/s).max(tolerance.relative){return Err(SolverError::NotSymmetric{row:i,column:j});}}}
        let mut l=ComplexMatrix::zeros(n,n)?;
        for j in 0..n{
            let mut d=a.data[j*n+j].re;for k in 0..j{let z=l.data[j*n+k];d-=z.re*z.re+z.im*z.im;}
            if !d.is_finite(){return Err(SolverError::Arithmetic("complex Cholesky pivot"));}
            if d<=0.0{return Err(SolverError::NotPositiveDefinite{index:j,pivot:d});}
            l.data[j*n+j]=Complex64::new(d.sqrt(),0.0);
            for i in j+1..n{let mut s=a.data[i*n+j];for k in 0..j{s=check(s-l.data[i*n+k]*l.data[j*n+k].conj())?;}
                l.data[i*n+j]=check(s/l.data[j*n+j])?;}
        }Ok(Self{lower:l})
    }
    pub fn lower(&self)->&ComplexMatrix{&self.lower}
    pub fn solve(&self,b:&ComplexMatrix)->Result<ComplexMatrix,SolverError>{
        let(n,r)=(self.lower.rows,b.cols);if b.rows!=n{return Err(SolverError::Shape("complex Cholesky RHS"));}let mut x=b.clone();
        for i in 0..n{for c in 0..r{let mut s=x.data[i*r+c];for j in 0..i{s=check(s-self.lower.data[i*n+j]*x.data[j*r+c])?;}x.data[i*r+c]=check(s/self.lower.data[i*n+i])?;}}
        for i in(0..n).rev(){for c in 0..r{let mut s=x.data[i*r+c];for j in i+1..n{s=check(s-self.lower.data[j*n+i].conj()*x.data[j*r+c])?;}x.data[i*r+c]=check(s/self.lower.data[i*n+i])?;}}Ok(x)
    }
}
#[derive(Clone,Debug)]pub struct ComplexQr{q:ComplexMatrix,r:ComplexMatrix,rank:usize}
impl ComplexQr{
    /// Unpivoted thin Householder QR, m>=n. Two-dimensional host data only.
    /// Rank is a diagonal diagnostic, not a rank-revealing decomposition.
    pub fn factor(a:&ComplexMatrix,tolerance:Tolerance)->Result<Self,SolverError>{
        let(m,n)=(a.rows,a.cols);if m<n{return Err(SolverError::Shape("complex QR requires m>=n"));}
        let cutoff=tolerance.threshold(a.max_abs())?;let mut r=a.clone();
        let mut reflectors:Vec<Vec<Complex64>>=Vec::new();
        for k in 0..n{
            let norm=crate::numerics::norm_iter((k..m).map(|i|r.data[i*n+k].abs()))?;
            let mut v=vec![Complex64::ZERO;m-k];
            if norm>0.0{
                let x0=r.data[k*n+k];let phase=if x0.abs()==0.0{Complex64::ONE}else{x0.scale(1.0/x0.abs())};
                for i in k..m{v[i-k]=r.data[i*n+k].scale(1.0/norm);}v[0]=v[0]+phase;
                let vn=crate::numerics::norm_iter(v.iter().map(|z|z.abs()))?;for z in &mut v{*z=z.scale(1.0/vn);}
                for j in k..n{
                    let mut d=Complex64::ZERO;for i in k..m{d=check(d+v[i-k].conj()*r.data[i*n+j])?;}
                    d=d.scale(2.0);for i in k..m{r.data[i*n+j]=check(r.data[i*n+j]-v[i-k]*d)?;}
                }
                for i in k+1..m{r.data[i*n+k]=Complex64::ZERO;}
            }reflectors.push(v);
        }
        let mut q=ComplexMatrix::zeros(m,n)?;for j in 0..n{q.data[j*n+j]=Complex64::ONE;}
        for k in(0..n).rev(){let v=&reflectors[k];for j in 0..n{
            let mut d=Complex64::ZERO;for i in k..m{d=check(d+v[i-k].conj()*q.data[i*n+j])?;}
            d=d.scale(2.0);for i in k..m{q.data[i*n+j]=check(q.data[i*n+j]-v[i-k]*d)?;}
        }}
        let mut thin=ComplexMatrix::zeros(n,n)?;for i in 0..n{for j in i..n{thin.data[i*n+j]=r.data[i*n+j];}}
        let rank=(0..n).filter(|&j|thin.data[j*n+j].abs()>cutoff).count();Ok(Self{q,r:thin,rank})
    }
    pub fn q(&self)->&ComplexMatrix{&self.q} pub fn r(&self)->&ComplexMatrix{&self.r}pub fn rank(&self)->usize{self.rank}
    pub fn least_squares(&self,b:&ComplexMatrix)->Result<ComplexMatrix,SolverError>{
        let(m,n,p)=(self.q.rows,self.q.cols,b.cols);if b.rows!=m{return Err(SolverError::Shape("complex QR RHS"));}
        if self.rank<n{return Err(SolverError::RankDeficient{rank:self.rank,columns:n});}
        let mut x=ComplexMatrix::zeros(n,p)?;for j in 0..n{for c in 0..p{
            let mut s=Complex64::ZERO;for i in 0..m{s=check(s+self.q.data[i*n+j].conj()*b.data[i*p+c])?;}x.data[j*p+c]=s;
        }}for i in(0..n).rev(){for c in 0..p{let mut s=x.data[i*p+c];for j in i+1..n{s=check(s-self.r.data[i*n+j]*x.data[j*p+c])?;}x.data[i*p+c]=check(s/self.r.data[i*n+i])?;}}Ok(x)
    }
}
