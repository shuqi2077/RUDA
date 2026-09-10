// SPDX-License-Identifier: Apache-2.0
//! Requires a real fixed-warp32 device; an unsupported device is a failure, not a skip.
#[path="warp_common/mod.rs"]mod common;
use common::*;
use rusolver::tensor::*;
#[test]
fn cholesky_warp_matches_serial_and_fp64_across_shapes(){
    for(batch,n,nr)in[(1,1,1),(3,2,3),(5,7,1),(65,8,8),(3,16,4),(2,31,3),(5,32,8)]{
        let c=Case::new(Kind::Cholesky,batch,n,nr,1.0);let(a,b)=c.upload();
        let serial=run(c.kind,false,&a,&b);let warp=run(c.kind,true,&a,&b);
        c.check(&serial);c.check(&warp);compare(&serial,&warp);bits_eq(&floats(&a),&c.av);bits_eq(&floats(&b),&c.bv);
    }
}
#[test]
fn lu_warp_matches_serial_pivots_and_fp64_across_shapes(){
    for(batch,n,nr)in[(1,1,8),(3,2,3),(65,7,1),(5,16,4),(2,31,2),(3,32,8)]{
        let c=Case::new(Kind::Lu,batch,n,nr,1.0);let(a,b)=c.upload();
        let serial=run(c.kind,false,&a,&b);let warp=run(c.kind,true,&a,&b);
        c.check(&serial);c.check(&warp);compare(&serial,&warp);bits_eq(&floats(&a),&c.av);bits_eq(&floats(&b),&c.bv);
    }
}
#[test]
fn warp_extreme_finite_scales_and_rhs_columns(){
    for kind in[Kind::Cholesky,Kind::Lu]{for scale in[1e-30f32,1e-15,1e15,1e30]{
        let c=Case::new(kind,3,8,3,scale);let(a,b)=c.upload();let r=run(kind,true,&a,&b);c.check(&r);
        compare(&r,&run(kind,false,&a,&b));
    }}
}
#[test]
fn cholesky_mixed_failures_are_warp_uniform_and_zeroed(){
    let av=vec![4.,1.,1.,3., 1.,2.,2.,1., 1.,2.,0.,1., f32::NAN,0.,0.,1., 0.,0.,0.,0.];
    let a=upload(av.clone(),[5,2,2]);let b=upload(vec![1.;10],[5,2,1]);
    let w=run(Kind::Cholesky,true,&a,&b);let s=run(Kind::Cholesky,false,&a,&b);
    assert_eq!(ints(&w.info),vec![0,2,-2,-1,1]);compare(&w,&s);
    assert!(floats(&w.factor)[4..].iter().all(|v|*v==0.));assert!(floats(&w.solution)[2..].iter().all(|v|*v==0.));bits_eq(&floats(&a),&av);
}
#[test]
fn lu_mixed_failures_do_not_strand_other_blocks(){
    let av=vec![0.,2.,1.,3., 1.,2.,2.,4., f32::INFINITY,0.,0.,1., 0.,0.,0.,0.];
    let a=upload(av.clone(),[4,2,2]);let b=upload(vec![1.;8],[4,2,1]);
    let w=run(Kind::Lu,true,&a,&b);let s=run(Kind::Lu,false,&a,&b);
    assert_eq!(ints(&w.info),vec![0,2,-1,1]);compare(&w,&s);
    assert!(floats(&w.factor)[4..].iter().all(|v|*v==0.));assert!(ints(w.pivots.as_ref().unwrap())[2..].iter().all(|v|*v == -1));bits_eq(&floats(&a),&av);
}
#[test]
fn lu_ties_choose_the_first_maximum_row(){
    let a=upload(vec![2.,1.,0., -2.,3.,1., 1.,0.,4.],[1,3,3]);let b=upload(vec![1.,2.,3.],[1,3,1]);
    let w=run(Kind::Lu,true,&a,&b);let s=run(Kind::Lu,false,&a,&b);compare(&w,&s);
    assert_eq!(ints(w.pivots.as_ref().unwrap())[0],0);
}
#[test]
fn explicit_shift_and_shift_overflow_preserve_status(){
    let a=upload(vec![0.;4],[1,2,2]);let b=upload(vec![2.,4.],[1,2,1]);
    let r=cholesky_solve_batched_warp(&a,&b,BatchedCholeskyOptions{diagonal_shift:2.,..Default::default()}).unwrap();
    r.check_status_sync().unwrap();close(&floats(&r.solution),&[1.,2.],1e-5);
    let a=upload(vec![f32::MAX],[1,1,1]);let b=upload(vec![1.],[1,1,1]);
    let r=cholesky_solve_batched_warp(&a,&b,BatchedCholeskyOptions{diagonal_shift:f32::MAX,..Default::default()}).unwrap();
    assert_eq!(ints(&r.info),vec![-3]);assert_eq!(floats(&r.solution),vec![0.]);
}
#[test]
fn rhs_nan_and_lu_rescaling_overflow_are_not_success(){
    let a=upload(vec![1e-30],[1,1,1]);let b=upload(vec![1e30],[1,1,1]);
    let w=run(Kind::Lu,true,&a,&b);assert_eq!(ints(&w.info),vec![-3]);compare(&w,&run(Kind::Lu,false,&a,&b));
    let b=upload(vec![f32::NAN],[1,1,1]);
    for kind in[Kind::Lu,Kind::Cholesky]{assert_eq!(ints(&run(kind,true,&a,&b).info),vec![-1]);}
}
#[test]
fn cutoff_and_nonfinite_input_precedence_match_baselines(){
    let a=upload(vec![1.,0.,0.,1e-7],[1,2,2]);let b=upload(vec![1.;2],[1,2,1]);
    let w=run(Kind::Lu,true,&a,&b);assert_eq!(ints(&w.info),vec![2]);compare(&w,&run(Kind::Lu,false,&a,&b));
    let a=upload(vec![0.;4],[1,2,2]);let b=upload(vec![f32::NAN,1.],[1,2,1]);
    assert_eq!(ints(&run(Kind::Lu,true,&a,&b).info),vec![-1]);
}
#[test]
fn invalid_layout_shape_and_options_are_rejected(){
    let a=upload(vec![1.;9],[1,3,3]);let b=upload(vec![1.;3],[1,3,1]);let mut transposed=a.clone();transposed.meta.swap(1,2);
    assert!(cholesky_solve_batched_warp(&transposed,&b,Default::default()).is_err());
    assert!(lu_solve_batched_warp(&transposed,&b,Default::default()).is_err());
    assert!(cholesky_solve_batched_warp(&a,&b,BatchedCholeskyOptions{diagonal_shift:-1.,..Default::default()}).is_err());
    assert!(lu_solve_batched_warp(&a,&b,BatchedLuOptions{pivot_relative_tolerance:f32::NAN,..Default::default()}).is_err());
    let big=upload(vec![1.;33*33],[1,33,33]);let rhs=upload(vec![1.;33],[1,33,1]);
    assert!(lu_solve_batched_warp(&big,&rhs,Default::default()).is_err());
    let wrong=upload(vec![1.;6],[2,3,1]);assert!(cholesky_solve_batched_warp(&a,&wrong,Default::default()).is_err());
}
#[test]
fn empty_batches_issue_no_work(){
    let a=upload(vec![],[0,32,32]);let b=upload(vec![],[0,32,8]);
    assert_eq!(cholesky_solve_batched_warp(&a,&b,Default::default()).unwrap().submitted_kernels,0);
    assert_eq!(lu_solve_batched_warp(&a,&b,Default::default()).unwrap().submitted_kernels,0);
}
#[test]
fn shared_readonly_input_alias_is_supported(){
    let a=upload(vec![4.],[1,1,1]);
    for kind in[Kind::Lu,Kind::Cholesky]{let r=run(kind,true,&a,&a);assert_eq!(ints(&r.info),vec![0]);close(&floats(&r.solution),&[1.],1e-5);}
    assert_eq!(floats(&a),vec![4.]);
}
#[test]
fn sanitizer_smoke_all_lanes_tails_and_failures(){
    for kind in[Kind::Lu,Kind::Cholesky]{for(n,r)in[(7,3),(32,8)]{
        let c=Case::new(kind,3,n,r,1.);let(a,b)=c.upload();c.check(&run(kind,true,&a,&b));
    }}
    cholesky_mixed_failures_are_warp_uniform_and_zeroed();lu_mixed_failures_do_not_strand_other_blocks();
    explicit_shift_and_shift_overflow_preserve_status();rhs_nan_and_lu_rescaling_overflow_are_not_success();
}
