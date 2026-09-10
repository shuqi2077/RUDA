// SPDX-License-Identifier: Apache-2.0
//! Opt-in real FP64 solver operations on the EXISTING Ruda autodiff graph.
//!
//! Explicitly HOST only: APIs take Autodiff<Host>, so a GPU tensor cannot silently
//! round-trip to CPU. First-order VJPs reuse rusolver::adjoint's saved forward
//! factors. No second tape engine. The existing Backward trait has no Result;
//! a numerical/readback error in backward panics with context (never zero grads).
use crate::{Autodiff,checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},
    grads::Gradients,ops::{Backward,Ops,OpsKind},tensor::AutodiffTensor};
use ruda_tensor::{api::Tensor,TensorPrimitive,TensorMetadata,TensorData,DType};
use ruda_tensor_host::{Host,HostTensor};
use rusolver::{Matrix,SolverError,Tolerance,CholeskyOptions,SvdOptions,EigenOptions};
use rusolver::adjoint::{self,SolvePullback,CholeskyPullback,SvdPullback,EigenPullback};
fn matrix(t:&HostTensor)->Result<Matrix,SolverError>{
    if t.dtype()!=DType::F64{return Err(SolverError::InvalidOption("solver_host requires explicit FP64 tensors"));}
    let shape=t.shape();if shape.len()!=2{return Err(SolverError::Shape("solver_host matrix rank"));}
    let v=t.clone().into_data().to_vec::<f64>().map_err(|_|SolverError::Operator("FP64 host data"))?;
    Matrix::new(shape[0],shape[1],v)
}
fn vector(t:HostTensor)->Result<Vec<f64>,SolverError>{
    if t.dtype()!=DType::F64||t.shape().len()!=1{return Err(SolverError::Shape("FP64 vector cotangent required"));}
    t.into_data().to_vec::<f64>().map_err(|_|SolverError::Operator("FP64 vector data"))
}
fn primitive(m:Matrix)->HostTensor{let shape=[m.rows(),m.columns()];HostTensor::from_data(TensorData::new(m.into_values(),shape))}
fn unpack<C:CheckpointStrategy,const D:usize>(t:Tensor<Autodiff<Host,C>,D>)->Result<AutodiffTensor<Host>,SolverError>{
    match t.into_primitive(){TensorPrimitive::Float(t)=>Ok(t),TensorPrimitive::QFloat(_)=>Err(SolverError::InvalidOption("quantized solver tensors unsupported"))}
}
#[derive(Debug)]struct SolveBackward;
impl Backward<Host,2> for SolveBackward{
    type State=SolvePullback;
    fn backward(self,ops:Ops<Self::State,2>,grads:&mut Gradients,_:&mut Checkpointer){
        let g=matrix(&grads.consume::<Host>(&ops.node)).expect("solver_host solve cotangent");
        let(da,db)=ops.state.backward(g.view()).expect("solver_host solve backward numerical failure");
        let[a,b]=ops.parents;if let Some(p)=a{grads.register::<Host>(p.id,primitive(da));}if let Some(p)=b{grads.register::<Host>(p.id,primitive(db));}
    }
}
/// Differentiable X=A^-1 B, multiple RHS, FP64 HOST. Singular inputs return Err.
pub fn solve_host<C:CheckpointStrategy>(a:Tensor<Autodiff<Host,C>,2>,b:Tensor<Autodiff<Host,C>,2>,tol:Tolerance)
->Result<Tensor<Autodiff<Host,C>,2>,SolverError>{
    let(a,b)=(unpack(a)?,unpack(b)?);let am=matrix(&a.primitive)?;let bm=matrix(&b.primitive)?;
    let(x,pb)=adjoint::solve_with_pullback(am.view(),bm.view(),tol)?;
    let prep=SolveBackward.prepare::<C>([a.node,b.node]).compute_bound().stateful();
    let out=match prep{OpsKind::Tracked(p)=>p.finish(pb,primitive(x)),OpsKind::UnTracked(p)=>p.finish(primitive(x))};
    Ok(Tensor::from_primitive(TensorPrimitive::Float(out)))
}
#[derive(Clone,Debug)]enum UnaryPullback{Cholesky(CholeskyPullback),SingularValues(SvdPullback),EigenValues(EigenPullback)}
#[derive(Debug)]struct UnaryBackward;
impl Backward<Host,1> for UnaryBackward{
    type State=UnaryPullback;
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer){
        let g=grads.consume::<Host>(&ops.node);
        let da=match ops.state{
            UnaryPullback::Cholesky(p)=>matrix(&g).and_then(|g|p.backward(g.view())),
            UnaryPullback::SingularValues(p)=>vector(g).and_then(|v|p.values_backward(&v)),
            UnaryPullback::EigenValues(p)=>vector(g).and_then(|v|p.backward(&v,None)),
        }.expect("solver_host backward: invalid cotangent or degenerate spectrum");
        if let Some(p)=ops.parents[0].as_ref(){grads.register::<Host>(p.id,primitive(da));}
    }
}
fn unary<C:CheckpointStrategy,const D:usize>(a:AutodiffTensor<Host>,state:UnaryPullback,out:HostTensor)->Tensor<Autodiff<Host,C>,D>{
    let prep=UnaryBackward.prepare::<C>([a.node]).compute_bound().stateful();
    let t=match prep{OpsKind::Tracked(p)=>p.finish(state,out),OpsKind::UnTracked(p)=>p.finish(out)};
    Tensor::from_primitive(TensorPrimitive::Float(t))
}
/// A=L L^T, returning a full lower-triangular matrix. Symmetric-input gradient.
pub fn cholesky_host<C:CheckpointStrategy>(a:Tensor<Autodiff<Host,C>,2>,options:CholeskyOptions)->Result<Tensor<Autodiff<Host,C>,2>,SolverError>{
    let a=unpack(a)?;let m=matrix(&a.primitive)?;let(l,pb)=adjoint::cholesky_with_pullback(m.view(),options)?;
    Ok(unary(a,UnaryPullback::Cholesky(pb),primitive(l)))
}
/// Descending singular values; backward requires nonzero, separated values.
/// Forward proactively validates this restriction when the input is tracked.
pub fn singular_values_host<C:CheckpointStrategy>(a:Tensor<Autodiff<Host,C>,2>,options:SvdOptions,gap_tolerance:f64)
->Result<Tensor<Autodiff<Host,C>,1>,SolverError>{
    let a=unpack(a)?;let m=matrix(&a.primitive)?;let(f,pb)=adjoint::svd_with_pullback(m.view(),options,gap_tolerance)?;
    if a.is_tracked(){pb.values_backward(&vec![0.0;f.singular_values.len()])?;}
    let n=f.singular_values.len();let out=HostTensor::from_data(TensorData::new(f.singular_values,[n]));
    Ok(unary(a,UnaryPullback::SingularValues(pb),out))
}
/// Ascending real symmetric eigenvalues; no eigenvector graph output yet.
pub fn symmetric_eigenvalues_host<C:CheckpointStrategy>(a:Tensor<Autodiff<Host,C>,2>,options:EigenOptions,gap_tolerance:f64)
->Result<Tensor<Autodiff<Host,C>,1>,SolverError>{
    let a=unpack(a)?;let m=matrix(&a.primitive)?;let(f,pb)=adjoint::eigen_with_pullback(m.view(),options,gap_tolerance)?;
    if a.is_tracked(){pb.backward(&vec![0.0;f.values.len()],None)?;}
    let n=f.values.len();let out=HostTensor::from_data(TensorData::new(f.values,[n]));Ok(unary(a,UnaryPullback::EigenValues(pb),out))
}
#[cfg(test)]mod tests{
    use super::*;
    type AD=Autodiff<Host>;
    fn input(v:Vec<f64>,shape:[usize;2])->Tensor<AD,2>{
        Tensor::from_primitive(TensorPrimitive::Float(AutodiffTensor::<Host>::new(HostTensor::from_data(TensorData::new(v,shape))).require_grad()))
    }
    fn values(t:Tensor<Host,2>)->Vec<f64>{t.into_primitive().tensor().into_data().to_vec::<f64>().unwrap()}
    #[test]fn solve_composes_with_existing_backward(){
        let a=input(vec![4.0,1.0,1.0,3.0],[2,2]);let b=input(vec![6.0,7.0],[2,1]);
        let x=solve_host(a.clone(),b.clone(),Tolerance::default()).unwrap();let loss=x.clone().mul(x).sum();let g=loss.backward();
        let db=values(b.grad(&g).unwrap());assert!((db[0]-2.0/11.0).abs()<1e-9);assert!((db[1]-14.0/11.0).abs()<1e-9);
        let da=values(a.grad(&g).unwrap());assert!((da[0]+2.0/11.0).abs()<1e-9);assert!((da[3]+28.0/11.0).abs()<1e-9);
    }
    #[test]fn singular_value_sum_uses_graph(){
        let a=input(vec![3.0,0.0,0.0,1.0],[2,2]);let s=singular_values_host(a.clone(),Default::default(),1e-12).unwrap();
        let g=s.sum().backward();let da=values(a.grad(&g).unwrap());assert!((da[0]-1.0).abs()<1e-9&&(da[3]-1.0).abs()<1e-9);
    }
    #[test]fn cholesky_graph_is_differentiable(){
        let a=input(vec![4.0,0.0,0.0,9.0],[2,2]);let l=cholesky_host(a.clone(),Default::default()).unwrap();
        let g=l.sum().backward();let da=values(a.grad(&g).unwrap());assert!((da[0]-0.25).abs()<1e-9);assert!((da[3]-1.0/6.0).abs()<1e-9);
    }
    #[test]fn degenerate_spectrum_rejected_before_backward(){
        let a=input(vec![1.0,0.0,0.0,1.0],[2,2]);assert!(singular_values_host(a,Default::default(),1e-12).is_err());
    }
    #[test]fn eigenvalue_sum_is_trace(){let a=input(vec![4.0,1.0,1.0,3.0],[2,2]);let y=symmetric_eigenvalues_host(a.clone(),Default::default(),1e-12).unwrap();
        let g=y.sum().backward();let da=values(a.grad(&g).unwrap());assert!((da[0]-1.0).abs()<1e-9&&(da[3]-1.0).abs()<1e-9&&da[1].abs()<1e-9);}
}
