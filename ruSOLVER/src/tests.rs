// SPDX-License-Identifier: Apache-2.0
use super::*;
fn matrix(rows: usize, cols: usize, data: &[f64])->Matrix{
    Matrix::new(rows, cols, data.to_vec()).unwrap()
}
fn close(x: &[f64], y: &[f64], tol: f64){
    assert_eq!(x.len(), y.len());
    for(i, (&x, &y))in x.iter().zip(y).enumerate(){
        assert!(x.is_finite()&&(x-y).abs()<=tol*(1.0+y.abs()), "i={i}: {x} != {y}");
    }
}
fn multiply(a: &Matrix, b: &Matrix)->Matrix{
    assert_eq!(a.columns(), b.rows());
    let mut data=vec![0.0; a.rows()*b.columns()];
    for i in 0..a.rows(){
        for j in 0..b.columns(){
            for k in 0..a.columns(){
                data[i*b.columns()+j]+=a.values()[i*a.columns()+k]*b.values()[k*b.columns()+j];
            }
        }
    }
    Matrix::new(a.rows(), b.columns(), data).unwrap()
}
fn spd()->Matrix{
    matrix(3, 3, &[4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0])
}
#[test]
fn matrix_rejects_invalid_dimensions_ranges_and_values(){
    assert!(Matrix::zeros(0, 1).is_err());
    assert!(Matrix::zeros(usize::MAX, 2).is_err());
    assert!(Matrix::new(2, 2, vec![1.0]).is_err());
    assert!(Matrix::new(1, 1, vec![f64::NAN]).is_err());
    assert!(MatrixView::strided(2, 2, &[0.0; 4], 0, 3, 1).is_err());
    assert!(MatrixView::strided(2, 2, &[0.0; 4], 0, usize::MAX, 1).is_err());
    assert!(MatrixView::strided(1, 1, &[1.0], 0, 0, 1).is_err());
}
#[test]
fn strided_view_and_transpose_only_read_reachable_entries(){
    let storage=[f64::NAN, 4.0, 1.0, f64::NAN, 2.0, 3.0];
    let view=MatrixView::strided(2, 2, &storage, 1, 3, 1).unwrap();
    close(view.to_owned().unwrap().values(), &[4.0, 1.0, 2.0, 3.0], 0.0);
    close(view.transpose().to_owned().unwrap().values(), &[4.0, 2.0, 1.0, 3.0], 0.0);
    assert_eq!(view.get(2, 0), None);
}
#[test]
fn scaled_norm_handles_extreme_values_and_empty(){
    for scale in [1e300, 1e-300]{
        let norm=l2_norm(&[scale, -scale]).unwrap();
        assert!((norm/scale-2.0f64.sqrt()).abs()<1e-14);
    }
    assert_eq!(l2_norm(&[]).unwrap(), 0.0);
    assert!(l2_norm(&[f64::INFINITY]).is_err());
    assert!(l2_norm(&[f64::MAX; 4]).is_err());
}
#[test]
fn lu_partial_pivot_reconstruction_and_multiple_rhs(){
    let a=matrix(3, 3, &[0.0, 2.0, 1.0, 1.0, -2.0, -3.0, 3.0, -1.0, 2.0]);
    let before=a.clone();
    let expected=matrix(3, 2, &[1.0, 2.0, 2.0, -1.0, -1.0, 3.0]);
    let b=multiply(&a, &expected);
    let lu=Lu::factor(a.view(), Default::default()).unwrap();
    close(lu.solve(b.view()).unwrap().values(), expected.values(), 1e-12);
    let mut pa=a.clone();
    for(k, &p)in lu.pivots().iter().enumerate(){
        for j in 0..3{
            pa.data.swap(k*3+j, p*3+j);
        }
    }
    close(multiply(&lu.lower().unwrap(), &lu.upper().unwrap()).values(), pa.values(), 1e-12);
    assert_eq!(a, before);
}
#[test]
fn lu_transpose_solve_reuses_factors(){
    let a=matrix(3, 3, &[0.0, 2.0, 1.0, 1.0, -2.0, -3.0, 3.0, -1.0, 2.0]);
    let x=matrix(3, 2, &[1.0, 4.0, 2.0, -1.0, 3.0, 2.0]);
    let b=multiply(&a.transpose().unwrap(), &x);
    let lu=Lu::factor(a.view(), Default::default()).unwrap();
    close(lu.solve_transpose(b.view()).unwrap().values(), x.values(), 1e-12);
}
#[test]
fn lu_inverse_and_logdet_do_not_multiply_large_diagonals(){
    let a=spd();
    let lu=Lu::factor(a.view(), Default::default()).unwrap();
    close(multiply(&a, &lu.inverse().unwrap()).values(), Matrix::identity(3).unwrap().values(), 1e-12);
    let huge=matrix(2, 2, &[1e200, 0.0, 0.0, -1e200]);
    let (sign, log)=Lu::factor(huge.view(), Default::default()).unwrap().slogdet().unwrap();
    assert_eq!(sign, -1);
    assert!((log-2.0*(1e200f64).ln()).abs()<1e-10);
}
#[test]
fn lu_singular_threshold_and_overflow_are_explicit(){
    let singular=matrix(2, 2, &[1.0, 2.0, 2.0, 4.0]);
    assert!(matches!(Lu::factor(singular.view(), Default::default()), Err(SolverError::Singular{
        ..
    })));
    let tiny=matrix(2, 2, &[1.0, 0.0, 0.0, 1e-20]);
    assert!(Lu::factor(tiny.view(), Default::default()).is_err());
    assert!(Lu::factor(tiny.view(), Tolerance::EXACT).is_ok());
    let overflow=matrix(2, 2, &[f64::MAX, f64::MAX, f64::MAX, -f64::MAX]);
    assert!(matches!(Lu::factor(overflow.view(), Tolerance::EXACT), Err(SolverError::Arithmetic(_))));
    assert!(Lu::factor(spd().view(), Tolerance{
        absolute: -1.0, relative: 0.0
    }).is_err());
}
#[test]
fn cholesky_reconstruct_solve_and_logdet(){
    let a=spd();
    let c=Cholesky::factor(a.view(), Default::default()).unwrap();
    close(multiply(c.lower(), &c.lower().transpose().unwrap()).values(), a.values(), 1e-12);
    let x=matrix(3, 2, &[1.0, 0.0, 2.0, 1.0, 3.0, -1.0]);
    let b=multiply(&a, &x);
    close(c.solve(b.view()).unwrap().values(), x.values(), 1e-12);
    assert!((c.log_determinant().unwrap()-Lu::factor(a.view(), Default::default()).unwrap().slogdet().unwrap().1).abs()<1e-12);
}
#[test]
fn cholesky_rejects_nonsymmetric_and_nonpositive_without_jitter(){
    assert!(matches!(Cholesky::factor(matrix(2, 2, &[1.0, 2.0, 0.0, 1.0]).view(), Default::default()), Err(SolverError::NotSymmetric{
        ..
    })));
    assert!(matches!(Cholesky::factor(matrix(2, 2, &[1.0, 2.0, 2.0, 1.0]).view(), Default::default()), Err(SolverError::NotPositiveDefinite{
        index: 1, ..
    })));
    assert!(Cholesky::factor(matrix(1, 1, &[0.0]).view(), Default::default()).is_err());
}
#[test]
fn cholesky_tiny_positive_pivot_is_not_silently_clamped(){
    let c=Cholesky::factor(matrix(1, 1, &[1e-300]).view(), Default::default()).unwrap();
    assert!((c.lower().values()[0]/1e-150-1.0).abs()<1e-14);
    close(c.solve(matrix(1, 1, &[1e-300]).view()).unwrap().values(), &[1.0], 1e-14);
}
#[test]
fn qr_pivoted_reconstruction_and_thin_orthogonality(){
    let a=matrix(4, 3, &[1.0, 3.0, 2.0, 2.0, -1.0, 1.0, 0.0, 5.0, 4.0, 3.0, 2.0, -1.0]);
    let qr=Qr::factor(a.view(), Default::default()).unwrap();
    let q=qr.q().unwrap();
    let r=qr.r().unwrap();
    let mut ap=a.clone();
    for i in 0..4{
        for j in 0..3{
            ap.data[i*3+j]=a.data[i*3+qr.permutation()[j]];
        }
    }
    close(multiply(&q, &r).values(), ap.values(), 1e-12);
    close(multiply(&q.transpose().unwrap(), &q).values(), Matrix::identity(3).unwrap().values(), 1e-12);
}
#[test]
fn qr_least_squares_uses_all_observations(){
    let a=matrix(4, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
    let b=matrix(4, 1, &[1.0, 3.0, 5.0, 7.0]);
    let fit=Qr::factor(a.view(), Default::default()).unwrap().least_squares(b.view()).unwrap();
    close(fit.solution.values(), &[1.0, 2.0], 1e-12);
    assert!(fit.residual_norms[0]<1e-12);
}
#[test]
fn qr_rank_deficiency_and_underdetermined_are_not_fake_solutions(){
    let a=matrix(3, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
    let qr=Qr::factor(a.view(), Default::default()).unwrap();
    assert_eq!(qr.rank(), 1);
    assert!(matches!(qr.least_squares(matrix(3, 1, &[1.0, 2.0, 3.0]).view()), Err(SolverError::RankDeficient{
        ..
    })));
    assert!(Qr::factor(matrix(1, 2, &[1.0, 2.0]).view(), Default::default()).is_err());
    assert_eq!(Qr::factor(Matrix::zeros(2, 2).unwrap().view(), Default::default()).unwrap().rank(), 0);
}
#[test]
fn symmetric_eigen_known_values_and_eigenvectors(){
    let a=matrix(2, 2, &[2.0, 1.0, 1.0, 2.0]);
    let e=symmetric_eigen(a.view(), Default::default()).unwrap();
    close(&e.values, &[1.0, 3.0], 1e-12);
    let av=multiply(&a, &e.vectors);
    for i in 0..2{
        for j in 0..2{
            assert!((av.data[i*2+j]-e.vectors.data[i*2+j]*e.values[j]).abs()<1e-12);
        }
    }
    close(multiply(&e.vectors.transpose().unwrap(), &e.vectors).values(), Matrix::identity(2).unwrap().values(), 1e-12);
}
#[test]
fn symmetric_eigen_scale_repeated_and_zero_matrices(){
    for scale in [1e200, 1e-200]{
        let e=symmetric_eigen(matrix(2, 2, &[2.0*scale, scale, scale, 2.0*scale]).view(), Default::default()).unwrap();
        close(&e.values.iter().map(|x|x/scale).collect::<Vec<_>>(), &[1.0, 3.0], 1e-11);
    }
    let e=symmetric_eigen(Matrix::identity(3).unwrap().view(), Default::default()).unwrap();
    assert_eq!(e.sweeps, 0);
    assert_eq!(e.values, vec![1.0; 3]);
    let z=symmetric_eigen(Matrix::zeros(3, 3).unwrap().view(), Default::default()).unwrap();
    assert_eq!(z.values, vec![0.0; 3]);
}
#[test]
fn eigen_budget_and_invalid_input_fail_explicitly(){
    let a=matrix(4, 4, &[4.0, 1.0, 2.0, 0.5, 1.0, 3.0, -1.0, 2.0, 2.0, -1.0, 6.0, 1.0, 0.5, 2.0, 1.0, 2.0]);
    let options=EigenOptions{
        max_sweeps: 1, tolerance: Tolerance{
            absolute: 0.0, relative: 1e-15
        }, ..Default::default()
    };
    assert!(matches!(symmetric_eigen(a.view(), options), Err(SolverError::NonConvergence{
        ..
    })));
    assert!(symmetric_eigen(a.view(), EigenOptions{
        max_sweeps: 0, ..Default::default()
    }).is_err());
}
#[test]
fn cg_diagonal_preconditioner_and_true_residual(){
    let a=matrix(3, 3, &[1e-4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e4]);
    let b=[1e-4, 2.0, 3e4];
    let m=JacobiPreconditioner::from_matrix(a.view()).unwrap();
    let r=conjugate_gradient(&a, &b, None, &m, Default::default()).unwrap();
    assert!(r.converged());
    assert_eq!(r.iterations, 1);
    close(&r.solution, &[1.0, 2.0, 3.0], 1e-12);
    assert!(r.residual_norm<1e-8);
}
#[test]
fn cg_initial_guess_zero_rhs_and_limits(){
    let a=spd();
    let m=IdentityPreconditioner::new(3).unwrap();
    let z=conjugate_gradient(&a, &[0.0; 3], None, &m, Default::default()).unwrap();
    assert_eq!(z.iterations, 0);
    assert!(z.converged());
    let b=[6.0, 10.0, 8.0];
    let r=conjugate_gradient(&a, &b, Some(&[1.0, 2.0, 3.0]), &m, Default::default()).unwrap();
    assert_eq!(r.iterations, 0);
    let r=conjugate_gradient(&a, &b, None, &m, CgOptions{
        max_iterations: 0, ..Default::default()
    }).unwrap();
    assert_eq!(r.status, IterativeStatus::MaxIterations);
    let r=conjugate_gradient(&a, &b, None, &m, CgOptions{
        max_iterations: 1, ..Default::default()
    }).unwrap();
    assert!(!r.converged());
    assert!(r.residual_norm>0.0);
}
struct Poisson{
    n: usize
}
impl LinearOperator for Poisson{
    fn dimension(&self)->usize{
        self.n
    }
    fn apply(&self, x: &[f64], out: &mut[f64])->Result<(), SolverError>{
        for i in 0..self.n{
            out[i]=2.0*x[i]-if i>0{
                x[i-1]
            } else{
                0.0
            }-if i+1<self.n{
                x[i+1]
            } else{
                0.0
            };
        }
        Ok(())
    }
}
#[test]
fn cg_matrix_free_poisson_and_residual_replacement(){
    let a=Poisson{
        n: 17
    };
    let r=conjugate_gradient(&a, &[1.0; 17], None, &IdentityPreconditioner::new(17).unwrap(), CgOptions{
        max_iterations: 1000, residual_recompute_interval: 12, ..Default::default()
    }).unwrap();
    assert!(r.converged());
    for(i, &x)in r.solution.iter().enumerate(){
        assert!((x-((i+1)*(17-i)) as f64/2.0).abs()<1e-7);
    }
}
#[test]
fn cg_nonpositive_curvature_and_incomplete_callback_are_errors(){
    let a=matrix(1, 1, &[-1.0]);
    assert!(matches!(conjugate_gradient(&a, &[1.0], None, &IdentityPreconditioner::new(1).unwrap(), Default::default()), Err(SolverError::Breakdown(_))));
    struct Bad;
    impl LinearOperator for Bad{
        fn dimension(&self)->usize{
            2
        }
        fn apply(&self, _: &[f64], out: &mut[f64])->Result<(), SolverError>{
            out[0]=0.0;
            Ok(())
        }
    }
    assert!(matches!(conjugate_gradient(&Bad, &[1.0; 2], None, &IdentityPreconditioner::new(2).unwrap(), Default::default()), Err(SolverError::NonFinite{
        ..
    })));
}
#[test]
fn residual_is_relative_to_rhs_not_claimed_forward_error(){
    let a=spd();
    assert_eq!(relative_residual(a.view(), &[1.0, 2.0, 3.0], &[6.0, 10.0, 8.0]).unwrap(), 0.0);
    assert_eq!(relative_residual(a.view(), &[0.0; 3], &[0.0; 3]).unwrap(), 0.0);
    assert!(relative_residual(a.view(), &[1.0; 2], &[1.0; 3]).is_err());
}
#[test]
fn all_factor_solves_reject_wrong_rhs(){
    let a=spd();
    let b=matrix(2, 1, &[1.0, 2.0]);
    assert!(Lu::factor(a.view(), Default::default()).unwrap().solve(b.view()).is_err());
    assert!(Cholesky::factor(a.view(), Default::default()).unwrap().solve(b.view()).is_err());
    assert!(Qr::factor(a.view(), Default::default()).unwrap().least_squares(b.view()).is_err());
}
// Independent NumPy/SciPy fixtures, generated by tools/science/oracle.py.
#[path="fixtures.rs"]mod fixtures;
