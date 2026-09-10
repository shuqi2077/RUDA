// SPDX-License-Identifier: Apache-2.0
use ruintegrate::*;
fn main()->Result<(),Box<dyn std::error::Error>>{
    let q=integrate_infinite(|x|Ok((-x*x).exp()),InfiniteInterval::WholeLine{split:0.},Default::default())?;
    if !q.converged(){return Err(format!("quadrature not converged: {:?}",q.status).into());}
    println!("Gaussian integral = {:.12}, estimated error = {:.3e}",q.integral,q.estimated_absolute_error);
    let mut jac=|_:f64,_:&[f64],j:&mut[f64]|{j[0]=-1000.;Ok(())};
    let s=solve_bdf1(|t,y,d|{d[0]=-1000.*(y[0]-t.cos())-t.sin();Ok(())},Some(&mut jac),0.,1.,&[1.],Default::default())?;
    if !s.ode.reached_end(){return Err(format!("stiff integration stopped: {:?}",s.ode.status).into());}
    println!("BDF1 y(1) = {:.9}, exact = {:.9}, steps={}, Newton={}",s.ode.state[0],1.0f64.cos(),s.ode.accepted_steps,s.newton_iterations);
    let e=solve_ivp_events(|_,y,d|{d[0]=y[1];d[1]=-y[0];Ok(())},0.,4.,&[1.,0.],
        Rk45Options{max_step:0.03,..Default::default()},|_,y,g|{g[0]=y[0];Ok(())},
        &[EventSpec{direction:EventDirection::Decreasing,terminal:true}],Default::default())?;
    println!("oscillator zero crossing at t={:.9}; status={:?}",e.ode.time,e.ode.status);Ok(())
}
