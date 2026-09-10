// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::adjoint::*;
use crate::complex::*;
use crate::sparse_direct::*;
fn mat(m:usize,n:usize,v:&[f64])->Matrix{Matrix::new(m,n,v.to_vec()).unwrap()}
fn close(a:&[f64],b:&[f64],tol:f64){assert_eq!(a.len(),b.len());for(i,(&a,&b))in a.iter().zip(b).enumerate(){assert!(a.is_finite()&&(a-b).abs()<=tol*(1.0+b.abs()),"{i}: {a} vs {b}");}}
fn prod(a:&Matrix,b:&Matrix)->Matrix{mm(a.view(),b.view()).unwrap()}
fn reconstruction(s:&Svd)->Matrix{let mut us=s.u.clone();for i in 0..us.rows{for j in 0..us.cols{us.data[i*us.cols+j]*=s.singular_values[j];}}prod(&us,&s.vt)}
#[test]fn svd_tall_wide_and_square_reconstruct(){for(m,n)in[(7,3),(3,7),(4,4),(1,5),(5,1)]{
    let a=Matrix::new(m,n,(0..m*n).map(|k|((k*17+3)%29)as f64/10.0-1.0).collect()).unwrap();
    let s=Svd::factor(a.view(),Default::default()).unwrap();close(reconstruction(&s).values(),a.values(),1e-10);
    let k=m.min(n);close(prod(&s.u.transpose().unwrap(),&s.u).values(),Matrix::identity(k).unwrap().values(),1e-9);
    close(prod(&s.vt,&s.vt.transpose().unwrap()).values(),Matrix::identity(k).unwrap().values(),1e-9);
    assert!(s.singular_values.windows(2).all(|w|w[0]>=w[1]));
}}
#[test]fn svd_zero_and_rank_deficient_have_orthonormal_completion(){for a in[Matrix::zeros(4,3).unwrap(),mat(4,3,&[1.,2.,3.,2.,4.,6.,3.,6.,9.,4.,8.,12.])]{
    let s=Svd::factor(a.view(),Default::default()).unwrap();assert!(s.rank<=1);close(reconstruction(&s).values(),a.values(),1e-9);
    close(prod(&s.u.transpose().unwrap(),&s.u).values(),Matrix::identity(3).unwrap().values(),1e-8);
}}
#[test]fn svd_extreme_uniform_scales(){for scale in[1e200,1e-200]{let a=mat(2,2,&[3.*scale,0.,0.,scale]);let s=Svd::factor(a.view(),Default::default()).unwrap();close(&s.singular_values.iter().map(|x|x/scale).collect::<Vec<_>>(),&[3.,1.],1e-12);}}
#[test]fn svd_minimum_norm_and_pseudoinverse(){let a=mat(2,3,&[1.,0.,0.,0.,2.,0.]);let s=Svd::factor(a.view(),Default::default()).unwrap();
    close(s.solve(mat(2,1,&[1.,4.]).view()).unwrap().values(),&[1.,2.,0.],1e-12);let pinv=s.pseudo_inverse().unwrap();close(prod(&prod(&a,&pinv),&a).values(),a.values(),1e-12);}
#[test]fn svd_options_and_rhs_rejected(){let a=Matrix::identity(2).unwrap();assert!(Svd::factor(a.view(),SvdOptions{max_sweeps:0,..Default::default()}).is_err());
    assert!(Svd::factor(a.view(),Default::default()).unwrap().solve(mat(1,1,&[1.]).view()).is_err());}
fn cmat(m:usize,n:usize,v:&[(f64,f64)])->ComplexMatrix{ComplexMatrix::new(m,n,v.iter().map(|&(r,i)|Complex64::new(r,i)).collect()).unwrap()}
fn cmul(a:&ComplexMatrix,b:&ComplexMatrix)->ComplexMatrix{let mut out=ComplexMatrix::zeros(a.rows(),b.columns()).unwrap();for i in 0..a.rows(){for j in 0..b.columns(){let mut s=Complex64::ZERO;for k in 0..a.columns(){s=s+a.data[i*a.cols+k]*b.data[k*b.cols+j];}out.data[i*b.cols+j]=s;}}out}
fn cclose(a:&ComplexMatrix,b:&ComplexMatrix,tol:f64){assert_eq!((a.rows,a.cols),(b.rows,b.cols));for(x,y)in a.data.iter().zip(&b.data){assert!((*x-*y).abs()<tol*(1.+y.abs()),"{x:?} != {y:?}");}}
#[test]fn complex_lu_pivots_multiple_rhs_and_adjoint(){let a=cmat(3,3,&[(0.,0.),(1.,2.),(3.,0.),(4.,-1.),(2.,1.),(0.,1.),(2.,0.),(3.,1.),(4.,-2.)]);
    let x=cmat(3,2,&[(1.,1.),(2.,0.),(-1.,2.),(3.,4.),(1.,-1.),(0.,2.)]);let f=ComplexLu::factor(&a,Default::default()).unwrap();
    cclose(&f.solve(&cmul(&a,&x)).unwrap(),&x,1e-10);cclose(&f.solve_adjoint(&cmul(&a.adjoint().unwrap(),&x)).unwrap(),&x,1e-10);assert_ne!(f.pivots()[0],0);}
#[test]fn complex_cholesky_hermitian_factor_and_solve(){let raw=cmat(3,2,&[(1.,1.),(2.,0.),(0.,2.),(3.,-1.),(2.,1.),(0.,-1.)]);let mut a=cmul(&raw,&raw.adjoint().unwrap());for i in 0..3{a.data[i*3+i]=a.data[i*3+i]+Complex64::new(2.,0.);}
    let f=ComplexCholesky::factor(&a,Default::default()).unwrap();cclose(&cmul(f.lower(),&f.lower().adjoint().unwrap()),&a,1e-10);
    let x=cmat(3,1,&[(1.,2.),(3.,-1.),(1.,0.)]);cclose(&f.solve(&cmul(&a,&x)).unwrap(),&x,1e-10);}
#[test]fn complex_qr_unitary_and_least_squares(){let a=cmat(4,2,&[(1.,1.),(2.,0.),(0.,2.),(3.,-1.),(2.,1.),(0.,-1.),(1.,0.),(-2.,1.)]);let f=ComplexQr::factor(&a,Default::default()).unwrap();
    cclose(&cmul(f.q(),f.r()),&a,1e-10);cclose(&cmul(&f.q().adjoint().unwrap(),f.q()),&ComplexMatrix::identity(2).unwrap(),1e-10);
    let x=cmat(2,1,&[(1.,2.),(-3.,1.)]);cclose(&f.least_squares(&cmul(&a,&x)).unwrap(),&x,1e-10);}
#[test]fn complex_rejects_invalid_singular_nonhermitian(){assert!(ComplexMatrix::new(1,1,vec![Complex64::new(f64::NAN,0.)]).is_err());
    let a=cmat(2,2,&[(1.,0.),(2.,0.),(2.,0.),(4.,0.)]);assert!(ComplexLu::factor(&a,Default::default()).is_err());
    let a=cmat(2,2,&[(2.,1.),(0.,0.),(0.,0.),(2.,0.)]);assert!(ComplexCholesky::factor(&a,Default::default()).is_err());}
#[test]fn sparse_lu_pivot_fill_transpose(){let a=mat(3,3,&[0.,2.,1.,1.,-2.,-3.,3.,-1.,2.]);let f=SparseLu::factor_csr(3,&[0,2,5,8],&[1,2,0,1,2,0,1,2],&[2.,1.,1.,-2.,-3.,3.,-1.,2.],Default::default()).unwrap();
    let x=mat(3,2,&[1.,2.,2.,-1.,-1.,3.]);close(f.solve(prod(&a,&x).view()).unwrap().values(),x.values(),1e-10);close(f.solve_transpose(prod(&a.transpose().unwrap(),&x).view()).unwrap().values(),x.values(),1e-10);}
#[test]fn sparse_lu_does_not_densify_tridiagonal(){let n=256;let(mut offsets,mut columns,mut values)=(vec![0],Vec::new(),Vec::new());for i in 0..n{if i>0{columns.push(i-1);values.push(-1.);}columns.push(i);values.push(3.);if i+1<n{columns.push(i+1);values.push(-1.);}offsets.push(values.len());}
    let f=SparseLu::factor_csr(n,&offsets,&columns,&values,Default::default()).unwrap();assert!(f.factor_nonzeros()<3*n);let mut b=vec![1.;n];b[0]=2.;b[n-1]=2.;close(f.solve(mat(n,1,&b).view()).unwrap().values(),&vec![1.;n],1e-10);}
#[test]fn sparse_duplicate_fill_budget_and_validation(){let f=SparseLu::factor_csr(2,&[0,3,5],&[0,1,0,0,1],&[2.,1.,2.,1.,3.],Default::default()).unwrap();close(f.solve(mat(2,1,&[6.,7.]).view()).unwrap().values(),&[1.,2.],1e-10);
    assert!(matches!(SparseLu::factor_csr(2,&[0,2,4],&[0,1,0,1],&[2.,1.,1.,2.],SparseLuOptions{max_factor_nonzeros:3,..Default::default()}),Err(SolverError::WorkspaceLimit{..})));
    assert!(SparseLu::factor_csr(2,&[0,3,2],&[0,1],&[1.,1.],Default::default()).is_err());}
fn loss(a:&Matrix,g:&Matrix)->f64{a.values().iter().zip(g.values()).map(|(a,b)|a*b).sum()}
fn finite_diff(a:&Matrix,mut f:impl FnMut(&Matrix)->f64)->Vec<f64>{let h=1e-6;(0..a.data.len()).map(|i|{let mut p=a.clone();let mut m=a.clone();p.data[i]+=h;m.data[i]-=h;(f(&p)-f(&m))/(2.*h)}).collect()}
#[test]fn solve_vjp_matches_finite_difference(){let a=mat(2,2,&[3.,1.,2.,4.]);let b=mat(2,2,&[1.,2.,3.,-1.]);let g=mat(2,2,&[0.2,0.4,-1.,2.]);let(_,pb)=solve_with_pullback(a.view(),b.view(),Default::default()).unwrap();let(da,db)=pb.backward(g.view()).unwrap();
    close(da.values(),&finite_diff(&a,|a|loss(&Lu::factor(a.view(),Default::default()).unwrap().solve(b.view()).unwrap(),&g)),2e-6);
    close(db.values(),&finite_diff(&b,|b|loss(&Lu::factor(a.view(),Default::default()).unwrap().solve(b.view()).unwrap(),&g)),2e-6);}
#[test]fn cholesky_symmetric_directional_vjp(){let a=mat(2,2,&[4.,1.,1.,3.]);let g=mat(2,2,&[1.,0.,2.,-0.3]);let(_,pb)=cholesky_with_pullback(a.view(),Default::default()).unwrap();let da=pb.backward(g.view()).unwrap();
    let direction=mat(2,2,&[0.5,0.2,0.2,-0.7]);let h=1e-6;let mut p=a.clone();let mut m=a.clone();for i in 0..4{p.data[i]+=h*direction.data[i];m.data[i]-=h*direction.data[i];}
    let fd=(loss(Cholesky::factor(p.view(),Default::default()).unwrap().lower(),&g)-loss(Cholesky::factor(m.view(),Default::default()).unwrap().lower(),&g))/(2.*h);
    assert!((fd-loss(&da,&direction)).abs()<1e-6);}
#[test]fn singular_value_vjp_and_degenerate_rejection(){let a=mat(3,2,&[3.,1.,0.,2.,1.,-1.]);let(_,pb)=svd_with_pullback(a.view(),Default::default(),1e-12).unwrap();let ds=[0.7,-0.2];let da=pb.values_backward(&ds).unwrap();
    close(da.values(),&finite_diff(&a,|a|Svd::factor(a.view(),Default::default()).unwrap().singular_values.iter().zip(ds).map(|(s,g)|s*g).sum()),2e-6);
    let(_,pb)=svd_with_pullback(Matrix::identity(2).unwrap().view(),Default::default(),1e-12).unwrap();assert!(pb.values_backward(&[1.,1.]).is_err());}
#[test]fn full_svd_vjp_handles_vector_signs_locally(){let a=mat(3,2,&[3.,1.,0.,2.,1.,-1.]);let(f,pb)=svd_with_pullback(a.view(),Default::default(),1e-12).unwrap();let gu=mat(3,2,&[0.1,0.2,0.4,-0.3,0.7,0.8]);let gv=mat(2,2,&[0.2,-0.4,0.3,0.1]);let ds=[0.3,0.8];
    let da=pb.backward(gu.view(),&ds,gv.view()).unwrap();let fd=finite_diff(&a,|a|{let mut s=Svd::factor(a.view(),Default::default()).unwrap();for j in 0..2{let align=(0..3).map(|i|s.u.data[i*2+j]*f.u.data[i*2+j]).sum::<f64>();if align<0.{for i in 0..3{s.u.data[i*2+j]*=-1.;}for i in 0..2{s.vt.data[j*2+i]*=-1.;}}}loss(&s.u,&gu)+loss(&s.vt,&gv)+s.singular_values.iter().zip(ds).map(|(s,g)|s*g).sum::<f64>()});close(da.values(),&fd,2e-5);}
#[test]fn eigen_value_vjp_on_symmetric_direction(){let a=mat(2,2,&[4.,1.,1.,3.]);let(_,pb)=eigen_with_pullback(a.view(),Default::default(),1e-12).unwrap();let da=pb.backward(&[0.3,0.7],None).unwrap();let h=1e-6;let mut p=a.clone();let mut m=a.clone();p.data[1]+=h;p.data[2]+=h;m.data[1]-=h;m.data[2]-=h;
    let scalar=|a:&Matrix|{let s=symmetric_eigen(a.view(),Default::default()).unwrap();0.3*s.values[0]+0.7*s.values[1]};assert!(((scalar(&p)-scalar(&m))/(2.*h)-(da.data[1]+da.data[2])).abs()<1e-6);}
#[test]fn qr_vjp_fixed_permutation(){let a=mat(3,2,&[4.,0.1,0.2,1.,1.,0.3]);let(qr,pb)=qr_with_pullback(a.view(),Default::default()).unwrap();let dq=mat(3,2,&[0.1,0.3,0.2,-0.7,0.5,0.4]);let dr=mat(2,2,&[0.2,-0.1,0.,0.4]);let da=pb.backward(dq.view(),dr.view()).unwrap();
    let fd=finite_diff(&a,|a|{let q=Qr::factor(a.view(),Default::default()).unwrap();assert_eq!(q.permutation(),qr.permutation());loss(&q.q().unwrap(),&dq)+loss(&q.r().unwrap(),&dr)});close(da.values(),&fd,2e-5);}
#[cfg(feature="sparse")]
#[test]fn sparse_direct_accepts_existing_one_based_format(){let a=rusparse::CsrMatrix::new(2,2,&[1,3,5],&[1,2,1,2],&[4.,1.,1.,3.],rusparse::IndexBase::One).unwrap();let f=SparseLu::from_rusparse(&a,Default::default()).unwrap();close(f.solve(mat(2,1,&[6.,7.]).view()).unwrap().values(),&[1.,2.],1e-10);}
#[path="distributed_tests.rs"]mod distributed_tests;
