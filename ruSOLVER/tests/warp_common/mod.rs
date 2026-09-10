// SPDX-License-Identifier: Apache-2.0
//! Real-device test/benchmark helpers, not a mocked runtime.
#![allow(dead_code)]
use ruda_core::{future::block_on,tensor::data::TensorData};
use ruda_driver_cuda::{CudaDevice,CudaRuntime};
use ruda_kernel::{dsl::Runtime,tensor::{RudaTensor,transfer::from_data,readback::into_data_sync}};
use rusolver::{Matrix,Lu,Cholesky};
use rusolver::tensor::*;
pub type Tensor=RudaTensor<CudaRuntime>;
#[derive(Clone,Copy,Debug)]pub enum Kind{Cholesky,Lu}
pub fn upload(v:Vec<f32>,s:[usize;3])->Tensor{from_data(TensorData::new(v,s),&CudaDevice::default())}
pub fn floats(t:&Tensor)->Vec<f32>{into_data_sync(t.clone()).to_vec::<f32>().expect("FP32 readback")}
pub fn ints(t:&Tensor)->Vec<i32>{into_data_sync(t.clone()).to_vec::<i32>().expect("I32 readback")}
pub fn sync(){block_on(CudaRuntime::client(&CudaDevice::default()).sync()).expect("CUDA synchronization failed");}
pub fn close(a:&[f32],b:&[f64],tol:f64){
    assert_eq!(a.len(),b.len());
    for(i,(&a,&b))in a.iter().zip(b).enumerate(){
        assert!(a.is_finite()&&b.is_finite()&&(a as f64-b).abs()<=tol*(1.0+b.abs()),"index {i}: {a} != {b}");
    }
}
pub fn bits_eq(a:&[f32],b:&[f32]){assert_eq!(a.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),b.iter().map(|v|v.to_bits()).collect::<Vec<_>>());}
pub struct Output{pub factor:Tensor,pub solution:Tensor,pub info:Tensor,pub pivots:Option<Tensor>,pub launches:usize}
pub fn run(kind:Kind,warp:bool,a:&Tensor,b:&Tensor)->Output{
    match kind{
        Kind::Cholesky=>{
            let r=if warp{cholesky_solve_batched_warp(a,b,Default::default())}else{cholesky_solve_batched(a,b,Default::default())}.expect("Cholesky submit");
            Output{factor:r.lower,solution:r.solution,info:r.info,pivots:None,launches:r.submitted_kernels}
        }
        Kind::Lu=>{
            let r=if warp{lu_solve_batched_warp(a,b,Default::default())}else{lu_solve_batched(a,b,Default::default())}.expect("LU submit");
            Output{factor:r.packed_lu,solution:r.solution,info:r.info,pivots:Some(r.pivots),launches:r.submitted_kernels}
        }
    }
}
pub struct Case{pub kind:Kind,pub batch:usize,pub n:usize,pub nr:usize,pub av:Vec<f32>,pub bv:Vec<f32>,pub expected:Vec<f64>}
impl Case{
    /// Diagonally dominant nonsymmetric matrices (with reversed rows) for LU,
    /// symmetric strictly diagonally dominant positive matrices for Cholesky.
    pub fn new(kind:Kind,batch:usize,n:usize,nr:usize,scale:f32)->Self{
        let mut av=vec![0f32;batch*n*n];let mut bv=vec![0f32;batch*n*nr];let mut expected=Vec::new();
        for s in 0..batch{
            let mut matrix=vec![0f32;n*n];
            for i in 0..n{for j in 0..n{
                let raw=match kind{
                    Kind::Cholesky=>((i.min(j)*7+i.max(j)*3+s)%11)as f32/100.0,
                    Kind::Lu=>((i*7+j*3+s)%13)as f32/100.0-0.06,
                };
                matrix[i*n+j]=scale*(raw+if i==j{n as f32+1.0}else{0.0});
            }}
            if matches!(kind,Kind::Lu){for i in 0..n/2{for j in 0..n{matrix.swap(i*n+j,(n-1-i)*n+j);}}}
            av[s*n*n..(s+1)*n*n].copy_from_slice(&matrix);
            for i in 0..n{for c in 0..nr{
                let mut value=0f64;
                for j in 0..n{let known=((j*3+c+s)%17)as f64/8.0-1.0;value+=matrix[i*n+j]as f64*known;}
                bv[s*n*nr+i*nr+c]=value as f32;
            }}
            let am=Matrix::from_f32(n,n,&matrix).unwrap();let bm=Matrix::from_f32(n,nr,&bv[s*n*nr..(s+1)*n*nr]).unwrap();
            let solved=match kind{
                Kind::Cholesky=>Cholesky::factor(am.view(),Default::default()).unwrap().solve(bm.view()).unwrap(),
                Kind::Lu=>Lu::factor(am.view(),Default::default()).unwrap().solve(bm.view()).unwrap(),
            };
            expected.extend_from_slice(solved.values());
        }
        Self{kind,batch,n,nr,av,bv,expected}
    }
    pub fn upload(&self)->(Tensor,Tensor){(upload(self.av.clone(),[self.batch,self.n,self.n]),upload(self.bv.clone(),[self.batch,self.n,self.nr]))}
    /// Independent FP64 solve and normalized residual/factor reconstruction.
    pub fn check(&self,r:&Output){
        assert_eq!(r.launches,usize::from(self.batch>0));assert_eq!(ints(&r.info),vec![0;self.batch]);
        let x=floats(&r.solution);let factor=floats(&r.factor);let piv=r.pivots.as_ref().map(ints);
        close(&x,&self.expected,8e-5);
        for s in 0..self.batch{
            let a=&self.av[s*self.n*self.n..(s+1)*self.n*self.n];
            let b=&self.bv[s*self.n*self.nr..(s+1)*self.n*self.nr];
            let xx=&x[s*self.n*self.nr..(s+1)*self.n*self.nr];
            let f=&factor[s*self.n*self.n..(s+1)*self.n*self.n];
            let mut residual=0f64;let mut norm_a=0f64;let mut norm_b=0f64;let mut norm_x=0f64;
            for i in 0..self.n{norm_a=norm_a.max((0..self.n).map(|j|(a[i*self.n+j]as f64).abs()).sum());}
            for i in 0..self.n{for c in 0..self.nr{
                let ax=(0..self.n).map(|j|a[i*self.n+j]as f64*xx[j*self.nr+c]as f64).sum::<f64>();
                residual=residual.max((ax-b[i*self.nr+c]as f64).abs());
                norm_b=norm_b.max((b[i*self.nr+c]as f64).abs());norm_x=norm_x.max((xx[i*self.nr+c]as f64).abs());
            }}
            assert!(residual/(norm_a*norm_x+norm_b).max(f64::MIN_POSITIVE)<2e-5,"solve residual");
            let mut pa=a.to_vec();
            if let Some(p)=&piv{for k in 0..self.n{let row=p[s*self.n+k];assert!(row>=k as i32&&row<self.n as i32);for j in 0..self.n{pa.swap(k*self.n+j,row as usize*self.n+j);}}}
            let scale=a.iter().map(|v|(*v as f64).abs()).fold(0f64,f64::max).max(f64::MIN_POSITIVE);
            for i in 0..self.n{for j in 0..self.n{
                let reconstructed=match self.kind{
                    Kind::Cholesky=>{
                        if i<j{assert_eq!(f[i*self.n+j],0.0);}
                        (0..=i.min(j)).map(|k|f[i*self.n+k]as f64*f[j*self.n+k]as f64).sum::<f64>()
                    }
                    Kind::Lu=>(0..self.n).map(|k|{
                        let l=if i==k{1.0}else if i>k{f[i*self.n+k]as f64}else{0.0};
                        let u=if k<=j{f[k*self.n+j]as f64}else{0.0};l*u
                    }).sum::<f64>(),
                };
                assert!(reconstructed.is_finite()&&(reconstructed-pa[i*self.n+j]as f64).abs()/scale<8e-5,"factor reconstruction at {s}/{i}/{j}");
            }}
        }
    }
}
pub fn compare(a:&Output,b:&Output){
    assert_eq!(ints(&a.info),ints(&b.info));
    close(&floats(&a.solution),&floats(&b.solution).into_iter().map(f64::from).collect::<Vec<_>>(),8e-5);
    // Relative factor checks are also independently scale-aware in Case::check.
    close(&floats(&a.factor),&floats(&b.factor).into_iter().map(f64::from).collect::<Vec<_>>(),8e-5);
    if let(Some(a),Some(b))=(&a.pivots,&b.pivots){assert_eq!(ints(a),ints(b));}
}
