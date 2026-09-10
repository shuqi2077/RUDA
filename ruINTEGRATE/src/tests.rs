// SPDX-License-Identifier: Apache-2.0
use super::*;
fn accurate()->QuadratureOptions{
    QuadratureOptions{
        absolute_tolerance: 1e-11, relative_tolerance: 1e-11, ..Default::default()
    }
}
#[test]
fn finite_quadrature_known_integrals(){
    let q=integrate(|x|Ok(x.powi(4)), 0.0, 1.0, accurate()).unwrap();
    assert!(q.converged());
    assert!((q.integral-0.2).abs()<1e-12);
    let q=integrate(|x|Ok(x.exp()), 0.0, 1.0, accurate()).unwrap();
    assert!(q.converged());
    assert!((q.integral-(std::f64::consts::E-1.0)).abs()<1e-11);
    let q=integrate(|x|Ok(x.sin()), 0.0, std::f64::consts::PI, accurate()).unwrap();
    assert!(q.converged());
    assert!((q.integral-2.0).abs()<1e-11);
    let q=integrate(|x|Ok((-x*x).exp()), -2.0, 2.0, accurate()).unwrap();
    assert!(q.converged());
    assert!((q.integral-1.764162781524843).abs()<1e-10);
}
#[test]
fn reversed_and_equal_bounds(){
    let q=integrate(|x|Ok(x*x), 1.0, 0.0, accurate()).unwrap();
    assert!((q.integral+1.0/3.0).abs()<1e-11);
    let q=integrate(|_|panic!("zero interval must not call function"), 2.0, 2.0, accurate()).unwrap();
    assert_eq!(q.evaluations, 0);
    assert_eq!(q.integral, 0.0);
}
#[test]
fn quadrature_cancellation_and_adaptive_refinement(){
    let q=integrate(|x|Ok(x*x*x), -1.0, 1.0, accurate()).unwrap();
    assert!(q.converged());
    assert!(q.integral.abs()<1e-12);
    let q=integrate(|x|Ok((x-0.13).abs()), 0.0, 1.0, accurate()).unwrap();
    assert!(q.converged());
    assert!((q.integral-(0.13f64.powi(2)+0.87f64.powi(2))/2.0).abs()<1e-10);
    assert!(q.intervals>1);
}
#[test]
fn quadrature_budget_exhaustion_is_not_success(){
    let q=integrate(|x|Ok((x-0.13).abs()), 0.0, 1.0, QuadratureOptions{
        max_intervals: 1, ..accurate()
    }).unwrap();
    assert_eq!(q.status, QuadratureStatus::MaxIntervals);
    assert_eq!(q.evaluations, 15);
    let q=integrate(|x|Ok((x-0.13).abs()), 0.0, 1.0, QuadratureOptions{
        max_evaluations: 15, ..accurate()
    }).unwrap();
    assert_eq!(q.status, QuadratureStatus::MaxEvaluations);
}
#[test]
fn quadrature_nonfinite_callback_and_invalid_options(){
    assert!(matches!(integrate(|_|Ok(f64::NAN), 0.0, 1.0, accurate()), Err(IntegrationError::NonFiniteFunction{
        ..
    })));
    assert!(matches!(integrate(|_|Err(IntegrationError::Callback("user")), 0.0, 1.0, accurate()), Err(IntegrationError::Callback("user"))));
    assert!(integrate(|x|Ok(x), 0.0, f64::INFINITY, accurate()).is_err());
    assert!(integrate(|x|Ok(x), 0.0, 1.0, QuadratureOptions{
        relative_tolerance: -1.0, ..accurate()
    }).is_err());
    assert!(integrate(|x|Ok(x), 0.0, 1.0, QuadratureOptions{
        max_evaluations: 14, ..accurate()
    }).is_err());
}
#[test]
fn quadrature_impossibly_tight_tolerance_never_claims_convergence(){
    let q=integrate(|_|Ok(1.0), 0.0, 1.0, QuadratureOptions{
        absolute_tolerance: 1e-30, relative_tolerance: 0.0, max_intervals: 8, ..Default::default()
    }).unwrap();
    assert!(!q.converged());
    assert!((q.integral-1.0).abs()<1e-12);
}
#[test]
fn ode_exponential_forward_and_backward(){
    let forward=solve_ivp(|_, y, d|{
        d[0]=-y[0];
        Ok(())
    }, 0.0, 5.0, &[1.0], Default::default()).unwrap();
    assert!(forward.reached_end());
    assert!((forward.state[0]-(-5.0f64).exp()).abs()<1e-8);
    let backward=solve_ivp(|_, y, d|{
        d[0]=y[0];
        Ok(())
    }, 1.0, 0.0, &[std::f64::consts::E], Default::default()).unwrap();
    assert!(backward.reached_end());
    assert!((backward.state[0]-1.0).abs()<1e-6);
}
#[test]
fn ode_harmonic_oscillator_and_fsal_counts(){
    let s=solve_ivp(|_, y, d|{
        d[0]=y[1];
        d[1]=-y[0];
        Ok(())
    }, 0.0, 2.0*std::f64::consts::PI, &[1.0, 0.0], Default::default()).unwrap();
    assert!(s.reached_end());
    assert!((s.state[0]-1.0).abs()<2e-6);
    assert!(s.state[1].abs()<2e-6);
    assert_eq!(s.evaluations, 1+6*(s.accepted_steps+s.rejected_steps));
    assert!(s.samples.is_empty());
}
#[test]
fn ode_zero_span_does_not_evaluate_callback(){
    let s=solve_ivp(|_, _, _|panic!("zero span"), 2.0, 2.0, &[4.0], Default::default()).unwrap();
    assert_eq!(s.state, vec![4.0]);
    assert_eq!(s.evaluations, 0);
    assert!(s.reached_end());
}
#[test]
fn ode_budgets_are_reported(){
    let f=|_: f64, y: &[f64], d: &mut[f64]|{
        d[0]=y[0];
        Ok(())
    };
    let s=solve_ivp(f, 0.0, 1.0, &[1.0], Rk45Options{
        max_steps: 0, ..Default::default()
    }).unwrap();
    assert_eq!(s.status, OdeStatus::MaxSteps);
    assert_eq!(s.evaluations, 0);
    let s=solve_ivp(f, 0.0, 1.0, &[1.0], Rk45Options{
        max_evaluations: 6, ..Default::default()
    }).unwrap();
    assert_eq!(s.status, OdeStatus::MaxEvaluations);
    assert_eq!(s.evaluations, 1);
}
#[test]
fn ode_rejected_step_leaves_committed_state_unchanged(){
    let s=solve_ivp(|_, y, d|{
        d[0]=100.0*y[0];
        Ok(())
    }, 0.0, 1.0, &[1.0], Rk45Options{
        initial_step: Some(1.0), max_steps: 1, ..Default::default()
    }).unwrap();
    assert_eq!(s.status, OdeStatus::MaxSteps);
    assert_eq!(s.rejected_steps, 1);
    assert_eq!(s.accepted_steps, 0);
    assert_eq!(s.time, 0.0);
    assert_eq!(s.state, vec![1.0]);
}
#[test]
fn ode_trajectory_limit_is_bounded(){
    let s=solve_ivp(|_, _, d|{
        d[0]=1.0;
        Ok(())
    }, 0.0, 1.0, &[0.0], Rk45Options{
        max_step: 0.01, save_trajectory: true, max_output_points: 3, ..Default::default()
    }).unwrap();
    assert_eq!(s.status, OdeStatus::OutputLimit);
    assert_eq!(s.samples.len(), 3);
    assert!(s.time<1.0);
    assert!((s.time-s.state[0]).abs()<1e-12);
}
#[test]
fn ode_min_step_and_partial_callback_are_detected(){
    let s=solve_ivp(|_, y, d|{
        d[0]=y[0];
        Ok(())
    }, 0.0, 1.0, &[1.0], Rk45Options{
        initial_step: Some(0.01), min_step: 0.1, ..Default::default()
    }).unwrap();
    assert_eq!(s.status, OdeStatus::StepTooSmall);
    assert!(matches!(solve_ivp(|_, _, d|{
        d[0]=1.0;
        Ok(())
    }, 0.0, 1.0, &[0.0, 0.0], Default::default()), Err(IntegrationError::NonFiniteFunction{
        index: 1, ..
    })));
}
#[test]
fn ode_callback_error_and_invalid_state(){
    assert!(matches!(solve_ivp(|_, _, _|Err(IntegrationError::Callback("failed")), 0.0, 1.0, &[0.0], Default::default()), Err(IntegrationError::Callback("failed"))));
    assert!(solve_ivp(|_, _, _|Ok(()), 0.0, 1.0, &[], Default::default()).is_err());
    assert!(solve_ivp(|_, _, _|Ok(()), 0.0, 1.0, &[f64::INFINITY], Default::default()).is_err());
    assert!(solve_ivp(|_, _, _|Ok(()), 0.0, 1.0, &[0.0], Rk45Options{
        initial_step: Some(-1.0), ..Default::default()
    }).is_err());
}
#[test]
fn ode_constant_derivative_and_input_unchanged(){
    let y=[1.0, 2.0];
    let s=solve_ivp(|_, _, d|{
        d[0]=2.0;
        d[1]=-1.0;
        Ok(())
    }, 0.0, 2.0, &y, Rk45Options{
        save_trajectory: true, ..Default::default()
    }).unwrap();
    assert!(s.reached_end());
    assert!((s.state[0]-5.0).abs()<1e-12);
    assert!(s.state[1].abs()<1e-12);
    assert_eq!(y, [1.0, 2.0]);
    assert_eq!(s.samples.last().unwrap().time, 2.0);
}
#[test]
fn quadrature_unrepresentable_half_width_is_not_zero_success(){
    let result=integrate(|_|Ok(1e308), 0.0, f64::from_bits(1), accurate());
    assert!(matches!(result, Err(IntegrationError::Arithmetic(_))));
}
