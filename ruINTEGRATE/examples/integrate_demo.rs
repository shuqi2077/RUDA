// SPDX-License-Identifier: Apache-2.0
use ruintegrate::{
    integrate, solve_ivp, Rk45Options
};
fn main()->Result<(), Box<dyn std::error::Error>>{
    let integral=integrate(|x|Ok((-x*x).exp()), -2.0, 2.0, Default::default())?;
    if !integral.converged(){
        return Err(format!("quadrature failed: {:?}", integral.status).into());
    }
    println!("Gaussian integral={}, estimated error={:e}, evaluations={}", integral.integral, integral.estimated_absolute_error, integral.evaluations);
    // Harmonic oscillator: position'=velocity, velocity'=-position.
    let solution=solve_ivp(|_, y, dy|{
        dy[0]=y[1];
        dy[1]=-y[0];
        Ok(())
    },
    0.0, 2.0*std::f64::consts::PI, &[1.0, 0.0], Rk45Options::default())?;
    if !solution.reached_end(){
        return Err(format!("ODE failed: {:?}", solution.status).into());
    }
    println!("oscillator y(2pi)={:?}, accepted={}, rejected={}", solution.state, solution.accepted_steps, solution.rejected_steps);
    assert!((solution.state[0]-1.0).abs()<1e-5);
    Ok(())
}
