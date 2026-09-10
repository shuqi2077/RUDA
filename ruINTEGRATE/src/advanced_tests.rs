// SPDX-License-Identifier: Apache-2.0
use super::*;
#[test]fn improper_exponential_gaussian_and_cauchy(){
    let q=integrate_infinite(|x|Ok((-x).exp()),InfiniteInterval::Above(0.),Default::default()).unwrap();assert!(q.converged());assert!((q.integral-1.).abs()<1e-8);
    let q=integrate_infinite(|x|Ok(x.exp()),InfiniteInterval::Below(0.),Default::default()).unwrap();assert!(q.converged());assert!((q.integral-1.).abs()<1e-8);
    let q=integrate_infinite(|x|Ok((-x*x).exp()),InfiniteInterval::WholeLine{split:0.},Default::default()).unwrap();assert!(q.converged());assert!((q.integral-std::f64::consts::PI.sqrt()).abs()<1e-8);
    let q=integrate_infinite(|x|Ok(1./(1.+x*x)),InfiniteInterval::WholeLine{split:0.},Default::default()).unwrap();assert!(q.converged());assert!((q.integral-std::f64::consts::PI).abs()<1e-8);
}
#[test]fn improper_rejects_invalid_and_reports_divergence(){assert!(integrate_infinite(|x|Ok(x),InfiniteInterval::Above(f64::NAN),Default::default()).is_err());
    let q=integrate_infinite(|_|Ok(1.),InfiniteInterval::Above(0.),QuadratureOptions{max_intervals:16,..Default::default()});if let Ok(q)=q{assert!(!q.converged());}}
#[test]fn wholeline_does_not_silently_compute_principal_value(){let q=integrate_infinite(|x|Ok(x/(1.+x*x)),InfiniteInterval::WholeLine{split:0.},QuadratureOptions{max_intervals:32,..Default::default()});if let Ok(q)=q{assert!(!q.converged());}}
#[test]fn stiff_decay_analytic_and_numerical_jacobian(){for analytic in[false,true]{let mut j=|_:f64,_:&[f64],out:&mut[f64]|{out[0]=-1000.;Ok(())};
    let opt=Bdf1Options{integration:Rk45Options{initial_step:Some(0.1),absolute_tolerance:1e-7,relative_tolerance:1e-4,..Default::default()},..Default::default()};
    let r=solve_bdf1(|_,y,d|{d[0]=-1000.*y[0];Ok(())},if analytic{Some(&mut j)}else{None},0.,1.,&[1.],opt).unwrap();
    assert!(r.ode.reached_end(),"{:?}",r.ode.status);assert!(r.ode.state[0].abs()<2e-6);assert!(r.factorizations>0&&r.newton_iterations>0);
}}
#[test]fn stiff_forced_solution_tracks_slow_mode(){let mut j=|_:f64,_:&[f64],out:&mut[f64]|{out[0]=-1000.;Ok(())};
    let r=solve_bdf1(|t,y,d|{d[0]=-1000.*(y[0]-t.cos())-t.sin();Ok(())},Some(&mut j),0.,1.,&[1.],Default::default()).unwrap();assert!(r.ode.reached_end());assert!((r.ode.state[0]-1.0f64.cos()).abs()<1e-4);}
#[test]fn stiff_robertson_mass_conservation(){let mut jac=|_:f64,y:&[f64],j:&mut[f64]|{j.copy_from_slice(&[-0.04,1e4*y[2],1e4*y[1],0.04,-1e4*y[2]-6e7*y[1],-1e4*y[1],0.,6e7*y[1],0.]);Ok(())};
    let r=solve_bdf1(|_,y,d|{d[0]=-0.04*y[0]+1e4*y[1]*y[2];d[2]=3e7*y[1]*y[1];d[1]=-d[0]-d[2];Ok(())},Some(&mut jac),0.,1.,&[1.,0.,0.],
        Bdf1Options{integration:Rk45Options{absolute_tolerance:1e-9,relative_tolerance:1e-5,..Default::default()},..Default::default()}).unwrap();
    assert!(r.ode.reached_end(),"{:?}",r.ode.status);assert!((r.ode.state.iter().sum::<f64>()-1.).abs()<1e-8);assert!((r.ode.state[0]-0.9664597).abs()<3e-4);
}
#[test]fn stiff_budget_returns_last_committed_state(){let r=solve_bdf1(|_,y,d|{d[0]=-1000.*y[0];Ok(())},None,0.,1.,&[1.],Bdf1Options{integration:Rk45Options{initial_step:Some(1.),max_steps:1,..Default::default()},..Default::default()}).unwrap();assert_eq!(r.ode.status,OdeStatus::MaxSteps);assert_eq!(r.ode.time,0.);assert_eq!(r.ode.state,vec![1.]);}
#[test]fn stiff_invalid_jacobian_and_dimension(){let mut j=|_:f64,_:&[f64],_:&mut[f64]|Ok(());assert!(solve_bdf1(|_,y,d|{d[0]=-y[0];Ok(())},Some(&mut j),0.,1.,&[1.],Default::default()).is_err());
    assert!(solve_bdf1(|_,_,_|Ok(()),None,0.,1.,&[1.;3],Bdf1Options{max_dimension:2,..Default::default()}).is_err());}
#[test]fn stiff_reverse_time(){let r=solve_bdf1(|_,y,d|{d[0]=y[0];Ok(())},None,1.,0.,&[std::f64::consts::E],Default::default()).unwrap();assert!(r.ode.reached_end());assert!((r.ode.state[0]-1.).abs()<0.003);}
#[test]fn events_terminal_and_nonterminal_order(){let specs=[EventSpec::default(),EventSpec{terminal:true,..Default::default()}];
    let r=solve_ivp_events(|_,_,d|{d[0]=1.;Ok(())},0.,1.,&[0.],Rk45Options{initial_step:Some(1.),..Default::default()},
        |_,y,g|{g[0]=y[0]-0.25;g[1]=y[0]-0.5;Ok(())},&specs,Default::default()).unwrap();
    assert_eq!(r.ode.status,OdeStatus::Event);assert_eq!(r.events.len(),2);assert_eq!(r.events[0].index,0);assert!((r.ode.time-0.5).abs()<1e-8);
}
#[test]fn event_direction_is_physical_time(){let spec=[EventSpec{direction:EventDirection::Increasing,terminal:true}];
    let r=solve_ivp_events(|_,_,d|{d[0]=1.;Ok(())},1.,0.,&[1.],Default::default(),|_,y,g|{g[0]=y[0]-0.5;Ok(())},&spec,Default::default()).unwrap();
    assert_eq!(r.ode.status,OdeStatus::Event);assert!((r.ode.time-0.5).abs()<1e-8);
    let spec=[EventSpec{direction:EventDirection::Decreasing,terminal:true}];let r=solve_ivp_events(|_,_,d|{d[0]=1.;Ok(())},0.,1.,&[0.],Default::default(),|_,y,g|{g[0]=y[0]-0.5;Ok(())},&spec,Default::default()).unwrap();assert!(r.ode.reached_end());assert!(r.events.is_empty());
}
#[test]fn initial_root_and_no_duplicate_endpoint(){let spec=[EventSpec{terminal:true,..Default::default()}];let r=solve_ivp_events(|_,_,_|panic!("initial terminal"),0.,1.,&[0.],Default::default(),|_,y,g|{g[0]=y[0];Ok(())},&spec,Default::default()).unwrap();assert_eq!(r.ode.time,0.);assert_eq!(r.ode.evaluations,0);
    let r=solve_ivp_events(|_,_,d|{d[0]=1.;Ok(())},0.,1.,&[0.],Rk45Options{max_step:0.1,..Default::default()},|_,y,g|{g[0]=y[0];Ok(())},&[EventSpec::default()],Default::default()).unwrap();assert_eq!(r.events.len(),1);
}
#[test]fn oscillator_event_interpolation(){let r=solve_ivp_events(|_,y,d|{d[0]=y[1];d[1]=-y[0];Ok(())},0.,4.,&[1.,0.],Rk45Options{max_step:0.03,absolute_tolerance:1e-11,relative_tolerance:1e-10,..Default::default()},
    |_,y,g|{g[0]=y[0];Ok(())},&[EventSpec{direction:EventDirection::Decreasing,terminal:true}],Default::default()).unwrap();assert!((r.ode.time-std::f64::consts::FRAC_PI_2).abs()<1e-7);}
#[test]fn event_failure_and_budget_are_explicit(){let r=solve_ivp_events(|_,_,d|{d[0]=1.;Ok(())},0.,1.,&[0.],Default::default(),|_,_,_|Ok(()),&[EventSpec::default()],Default::default());assert!(r.is_err());
    let r=solve_ivp_events(|_,_,d|{d[0]=1.;Ok(())},0.,1.,&[0.],Default::default(),|_,y,g|{g[0]=y[0]-0.5;Ok(())},&[EventSpec::default()],EventOptions{max_event_evaluations:1,..Default::default()});assert!(r.is_err());}
#[test]fn stiff_event_location(){let r=solve_bdf1_events(|_,y,d|{d[0]=-1000.*y[0];Ok(())},None,0.,0.1,&[1.],Bdf1Options{integration:Rk45Options{absolute_tolerance:1e-8,relative_tolerance:1e-5,..Default::default()},..Default::default()},
    |_,y,g|{g[0]=y[0]-0.1;Ok(())},&[EventSpec{direction:EventDirection::Decreasing,terminal:true}],Default::default()).unwrap();
    assert_eq!(r.stiff.ode.status,OdeStatus::Event);assert!((r.stiff.ode.time-10.0f64.ln()/1000.).abs()<2e-5);
}
