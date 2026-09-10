// SPDX-License-Identifier: Apache-2.0
//! Infinite-domain quadrature by rational variable transforms and existing GK15.
//! Not a Cauchy principal-value or oscillatory-tail algorithm. Error is estimated,
//! not a proof of absolute convergence. Interior singularities need explicit splits.
use crate::{integrate,IntegrationError,QuadratureOptions,QuadratureReport};
use crate::error::checked;
#[derive(Clone,Copy,Debug)]pub enum InfiniteInterval{Above(f64),Below(f64),WholeLine{split:f64}}
pub fn integrate_infinite(mut f:impl FnMut(f64)->Result<f64,IntegrationError>,interval:InfiniteInterval,
options:QuadratureOptions)->Result<QuadratureReport,IntegrationError>{
    match interval{
        InfiniteInterval::Above(a)|InfiniteInterval::Below(a)=>{
            if !a.is_finite(){return Err(IntegrationError::NonFiniteInput("infinite interval endpoint"));}
            tail(&mut f,a,matches!(interval,InfiniteInterval::Above(_)),options)
        }
        InfiniteInterval::WholeLine{split}=>{
            if !split.is_finite(){return Err(IntegrationError::NonFiniteInput("infinite split"));}
            // Integrate tails independently, not f(split+x)+f(split-x) at the same
            // nodes: cancellation must not disguise two divergent one-sided tails.
            if options.max_evaluations<30||options.max_intervals<2{return Err(IntegrationError::InvalidOption("whole-line integration needs two tail budgets"));}
            let half=QuadratureOptions{absolute_tolerance:options.absolute_tolerance*0.5,
                max_evaluations:options.max_evaluations/2,max_intervals:options.max_intervals/2,..options};
            let left=tail(&mut f,split,false,half)?;
            let right=tail(&mut f,split,true,half)?;
            let integral=checked(left.integral+right.integral,"sum of improper tails")?;
            let error=checked(left.estimated_absolute_error+right.estimated_absolute_error,"improper error")?;
            let mut status=if !left.converged(){left.status}else{right.status};
            if status==crate::QuadratureStatus::Converged&&error>options.absolute_tolerance.max(options.relative_tolerance*integral.abs()){
                status=crate::QuadratureStatus::RoundoffLimit;
            }
            Ok(QuadratureReport{integral,estimated_absolute_error:error,evaluations:left.evaluations+right.evaluations,
                intervals:left.intervals+right.intervals,status})
        }
    }
}

fn tail(f:&mut impl FnMut(f64)->Result<f64,IntegrationError>,a:f64,above:bool,options:QuadratureOptions)->Result<QuadratureReport,IntegrationError>{
    let sign=if above{1.0}else{-1.0};
    integrate(|t|{let u=1.0-t;let x=checked(a+sign*t/u,"infinite transform coordinate")?;
        let y=f(x)?;checked((y/u)/u,"infinite transform Jacobian")},0.0,1.0,options)
}
