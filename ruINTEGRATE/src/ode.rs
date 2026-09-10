// SPDX-License-Identifier: Apache-2.0
use crate::IntegrationError;
use crate::error::{
    checked, validate_tolerance, vector
};
#[derive(Clone, Copy, Debug)]
pub struct Rk45Options{
    pub absolute_tolerance: f64, pub relative_tolerance: f64,
    /// Positive magnitude; direction is derived from t_start/t_end.
    pub initial_step: Option<f64>, pub min_step: f64, pub max_step: f64,
    /// Includes accepted AND rejected attempts.
    pub max_steps: usize, pub max_evaluations: usize,
    /// Default false: retain only final state, avoiding trajectory-size surprises.
    pub save_trajectory: bool, pub max_output_points: usize,
}
impl Default for Rk45Options{
    fn default()->Self{
        Self{
            absolute_tolerance: 1e-9, relative_tolerance: 1e-7, initial_step: None, min_step: 0.0, max_step: f64::INFINITY,
            max_steps: 100_000, max_evaluations: 600_001, save_trajectory: false, max_output_points: 10_000
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OdeStatus{
    ReachedEnd, Event, MaxSteps, MaxEvaluations, StepTooSmall, OutputLimit
}
#[derive(Clone, Debug)]
pub struct OdeSample{
    pub time: f64, pub state: Vec<f64>
}
#[derive(Clone, Debug)]
pub struct OdeReport{
    pub time: f64, pub state: Vec<f64>, pub accepted_steps: usize, pub rejected_steps: usize,
    pub evaluations: usize, pub status: OdeStatus, pub samples: Vec<OdeSample>,
}
impl OdeReport{
    pub fn reached_end(&self)->bool{
        self.status==OdeStatus::ReachedEnd
    }
}
// Dormand-Prince 5(4) Butcher tableau: mathematical constants, not copied source.
const C: [f64; 7]=[0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0];
const A: [[f64; 7]; 7]=[
[0.0; 7],
[1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
[19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
[9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
[35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0],
];
const B4: [f64; 7]=[5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0];
fn evaluate<F: FnMut(f64, &[f64], &mut[f64])->Result<(), IntegrationError>>(
f: &mut F, t: f64, y: &[f64], out: &mut[f64])->Result<(), IntegrationError>{
    out.fill(f64::NAN);
    f(t, y, out)?;
    for(index, &v)in out.iter().enumerate(){
        if !v.is_finite(){
            return Err(IntegrationError::NonFiniteFunction{
                at: t, index
            });
        }
    }
    Ok(())
}
fn state_copy(y: &[f64])->Result<Vec<f64>, IntegrationError>{
    let mut out=vector(y.len())?;
    out.copy_from_slice(y);
    Ok(out)
}
fn record(samples: &mut Vec<OdeSample>, time: f64, state: &[f64])->Result<(), IntegrationError>{
    samples.try_reserve(1).map_err(|_|IntegrationError::Allocation)?;
    samples.push(OdeSample{
        time, state: state_copy(state)?
    });
    Ok(())
}
/// Integrate y'=f(t,y) with an explicit adaptive Dormand-Prince 5(4) method.
/// Callback MUST overwrite every derivative element and be a deterministic function
/// of (t,y); it may be called at rejected/trial states. Callback errors propagate.
///
/// Error control uses max_i |err_i|/(atol+rtol*max(|y_i|,|y_next_i|)).
/// Derivative buffers are reused (six evaluations per attempted step after the first).
/// No stiff/implicit method, event detection, interpolation, automatic Jacobian or
/// autodiff is provided. Returned errors are local estimates, not global guarantees.
pub fn solve_ivp(f: impl FnMut(f64, &[f64], &mut[f64])->Result<(), IntegrationError>,
t_start: f64, t_end: f64, y0: &[f64], options: Rk45Options)->Result<OdeReport, IntegrationError>{
    solve_ivp_observed(f,t_start,t_end,y0,options,|_,_,_,_,_,_|Ok(None))
}
pub(crate) fn solve_ivp_observed(mut f: impl FnMut(f64, &[f64], &mut[f64])->Result<(), IntegrationError>,
t_start: f64, t_end: f64, y0: &[f64], options: Rk45Options,
mut observer: impl FnMut(f64,&[f64],&[f64],f64,&[f64],&[f64])->Result<Option<OdeSample>,IntegrationError>)
->Result<OdeReport, IntegrationError>{
    validate_tolerance(options.absolute_tolerance, options.relative_tolerance)?;
    if !t_start.is_finite()||!t_end.is_finite()||!(t_end-t_start).is_finite(){
        return Err(IntegrationError::NonFiniteInput("time interval"));
    }
    if y0.is_empty()||y0.iter().any(|x|!x.is_finite()){
        return Err(IntegrationError::NonFiniteInput("initial state must be nonempty and finite"));
    }
    if !options.min_step.is_finite()||options.min_step<0.0||options.max_step.is_nan()
    ||options.max_step<=0.0||options.max_step<options.min_step
    ||options.initial_step.is_some_and(|h|!h.is_finite()||h<=0.0)
    ||options.max_output_points==0{
        return Err(IntegrationError::InvalidOption("step bounds/initial step/output budget"));
    }
    let mut report=OdeReport{
        time: t_start, state: state_copy(y0)?, accepted_steps: 0, rejected_steps: 0,
        evaluations: 0, status: OdeStatus::ReachedEnd, samples: Vec::new()
    };
    if options.save_trajectory{
        record(&mut report.samples, t_start, y0)?;
    }
    if t_start==t_end{
        return Ok(report);
    }
    if options.max_steps==0{
        report.status=OdeStatus::MaxSteps;
        return Ok(report);
    }
    if options.max_evaluations<1{
        report.status=OdeStatus::MaxEvaluations;
        return Ok(report);
    }
    let n=y0.len();
    let mut k=[vector(n)?, vector(n)?, vector(n)?, vector(n)?, vector(n)?, vector(n)?, vector(n)?];
    let mut temp=vector(n)?;
    let mut candidate=vector(n)?;
    evaluate(&mut f, report.time, &report.state, &mut k[0])?;
    report.evaluations=1;
    let direction=if t_end>t_start{
        1.0
    } else{
        -1.0
    };
    let span=(t_end-t_start).abs();
    let mut h_abs=options.initial_step.unwrap_or(span*0.01).min(options.max_step).min(span);
    if h_abs==0.0{
        report.status=OdeStatus::StepTooSmall;
        return Ok(report);
    }
    let mut previous_rejected=false;
    for _ in 0..options.max_steps{
        if options.save_trajectory&&report.samples.len()>=options.max_output_points{
            report.status=OdeStatus::OutputLimit;
            return Ok(report);
        }
        let remaining=(t_end-report.time).abs();
        h_abs=h_abs.min(options.max_step).min(remaining);
        if h_abs<options.min_step&&h_abs<remaining{
            report.status=OdeStatus::StepTooSmall;
            return Ok(report);
        }
        let t_next=if h_abs==remaining{
            t_end
        } else{
            report.time+direction*h_abs
        };
        let h=t_next-report.time;
        if t_next==report.time{
            report.status=OdeStatus::StepTooSmall;
            return Ok(report);
        }
        if options.max_evaluations.saturating_sub(report.evaluations)<6{
            report.status=OdeStatus::MaxEvaluations;
            return Ok(report);
        }
        for stage in 1..7{
            for i in 0..n{
                let mut sum=0.0;
                for j in 0..stage{
                    sum=checked(sum+A[stage][j]*k[j][i], "ODE stage derivative sum")?;
                }
                temp[i]=checked(h.mul_add(sum, report.state[i]), "ODE stage state")?;
            }
            let time=if stage>=5{
                t_next
            } else{
                report.time+h*C[stage]
            };
            evaluate(&mut f, time, &temp, &mut k[stage])?;
            report.evaluations+=1;
            if stage==6{
                candidate.copy_from_slice(&temp);
            }
        }
        let mut error_norm=0.0f64;
        for i in 0..n{
            let mut error=0.0;
            for j in 0..7{
                error=checked(error+(A[6][j]-B4[j])*k[j][i], "ODE embedded error sum")?;
            }
            let error=checked(h*error, "ODE embedded error")?.abs();
            let scale=checked(options.absolute_tolerance+options.relative_tolerance*report.state[i].abs().max(candidate[i].abs()), "ODE error scale")?;
            let ratio=if scale==0.0{
                if error==0.0{
                    0.0
                } else{
                    f64::INFINITY
                }
            } else{
                error/scale
            };
            error_norm=error_norm.max(ratio);
        }
        if error_norm<=1.0{
            if let Some(stop)=observer(report.time,&report.state,&k[0],t_next,&candidate,&k[6])? {
                report.time=stop.time;report.state=stop.state;report.accepted_steps+=1;
                report.status=OdeStatus::Event;
                if options.save_trajectory { record(&mut report.samples,report.time,&report.state)?; }
                return Ok(report);
            }
            report.time=t_next;
            report.state.copy_from_slice(&candidate);
            report.accepted_steps+=1;
            if options.save_trajectory{
                record(&mut report.samples, report.time, &report.state)?;
            }
            if t_next==t_end{
                report.status=OdeStatus::ReachedEnd;
                return Ok(report);
            }
            // Reuse the derivative at the accepted endpoint (FSAL).
            let (first, last)=k.split_at_mut(6);
            first[0].copy_from_slice(&last[0]);
            let mut factor=if error_norm==0.0{
                5.0
            } else{
                (0.9*error_norm.powf(-0.2)).clamp(0.2, 5.0)
            };
            if previous_rejected{
                factor=factor.min(1.0);
            }
            h_abs=(h.abs()*factor).min(options.max_step);
            previous_rejected=false;
        } else{
            report.rejected_steps+=1;
            h_abs=h.abs()*(0.9*error_norm.powf(-0.2)).clamp(0.2, 0.9);
            previous_rejected=true;
        }
    }
    report.status=OdeStatus::MaxSteps;
    Ok(report)
}
