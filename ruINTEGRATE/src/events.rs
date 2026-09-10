// SPDX-License-Identifier: Apache-2.0
//! Sign-change event location on accepted steps, with cubic Hermite interpolation
//! from endpoint states/derivatives. Root tolerance is NOT an ODE accuracy bound.
//! Tangencies and multiple roots within one step may be missed; bound max_step.
use crate::{IntegrationError,Rk45Options,OdeReport,OdeSample,OdeStatus,Bdf1Options,StiffReport,Jacobian};
#[derive(Clone,Copy,Debug,PartialEq,Eq)]pub enum EventDirection{Any,Increasing,Decreasing}
#[derive(Clone,Copy,Debug)]pub struct EventSpec{pub direction:EventDirection,pub terminal:bool}
impl Default for EventSpec{fn default()->Self{Self{direction:EventDirection::Any,terminal:false}}}
#[derive(Clone,Copy,Debug)]pub struct EventOptions{pub absolute_time_tolerance:f64,pub relative_time_tolerance:f64,
    pub max_root_iterations:usize,pub max_event_evaluations:usize,pub max_events:usize}
impl Default for EventOptions{fn default()->Self{Self{absolute_time_tolerance:1e-10,relative_time_tolerance:1e-12,
    max_root_iterations:100,max_event_evaluations:100_000,max_events:10_000}}}
#[derive(Clone,Debug)]pub struct EventOccurrence{pub index:usize,pub time:f64,pub state:Vec<f64>,pub value:f64}
#[derive(Clone,Debug)]pub struct EventReport{pub ode:OdeReport,pub events:Vec<EventOccurrence>,pub event_evaluations:usize}
#[derive(Clone,Debug)]pub struct StiffEventReport{pub stiff:StiffReport,pub events:Vec<EventOccurrence>,pub event_evaluations:usize}
struct Tracker<'a,E>{event:E,specs:&'a[EventSpec],options:EventOptions,previous:Vec<f64>,last:Vec<Option<f64>>,hits:Vec<EventOccurrence>,eval:usize}
impl<'a,E:FnMut(f64,&[f64],&mut[f64])->Result<(),IntegrationError>>Tracker<'a,E>{
    fn new(event:E,specs:&'a[EventSpec],options:EventOptions,t:f64,y:&[f64])->Result<Self,IntegrationError>{
        if specs.is_empty()||!options.absolute_time_tolerance.is_finite()||!options.relative_time_tolerance.is_finite()
            ||options.absolute_time_tolerance<=0.0||options.relative_time_tolerance<0.0||options.max_root_iterations==0||options.max_events==0{
            return Err(IntegrationError::InvalidOption("event specifications/tolerance/budget"));}
        let mut s=Self{event,specs,options,previous:Vec::new(),last:vec![None;specs.len()],hits:Vec::new(),eval:0};
        s.previous=s.evaluate(t,y)?;Ok(s)
    }
    fn evaluate(&mut self,t:f64,y:&[f64])->Result<Vec<f64>,IntegrationError>{
        if self.eval>=self.options.max_event_evaluations{return Err(IntegrationError::Callback("event evaluation budget exhausted"));}
        let mut out=vec![f64::NAN;self.specs.len()];(self.event)(t,y,&mut out)?;self.eval+=1;
        for(index,&v)in out.iter().enumerate(){if !v.is_finite(){return Err(IntegrationError::NonFiniteFunction{at:t,index});}}Ok(out)
    }
    fn initial(&mut self,t:f64,y:&[f64])->Result<bool,IntegrationError>{
        let mut stop=false;for i in 0..self.specs.len(){if self.previous[i]==0.0{
            self.push(EventOccurrence{index:i,time:t,state:y.to_vec(),value:0.0})?;stop|=self.specs[i].terminal;
        }}Ok(stop)
    }
    fn push(&mut self,hit:EventOccurrence)->Result<(),IntegrationError>{
        if self.hits.len()>=self.options.max_events{return Err(IntegrationError::Callback("event output budget exhausted"));}
        self.last[hit.index]=Some(hit.time);self.hits.push(hit);Ok(())
    }
    fn step(&mut self,t0:f64,y0:&[f64],f0:&[f64],t1:f64,y1:&[f64],f1:&[f64])->Result<Option<OdeSample>,IntegrationError>{
        let current=self.evaluate(t1,y1)?;let mut hits=Vec::new();
        for i in 0..self.specs.len(){
            let(g0,g1)=(self.previous[i],current[i]);
            // A zero at the start was reported on the previous step. Don't report
            // it again; require leaving zero and a subsequent new crossing.
            let crosses=g0!=0.0&&(g1==0.0||g0.signum()!=g1.signum());
            let (low,high)=if t1>t0{(g0,g1)}else{(g1,g0)};
            let direction=match self.specs[i].direction{EventDirection::Any=>true,EventDirection::Increasing=>high>low,EventDirection::Decreasing=>high<low};
            if !crosses||!direction{continue;}
            let(mut lo,mut hi,mut gl)=(0.0,1.0,g0);let mut hit=None;
            if g1==0.0{hit=Some(EventOccurrence{index:i,time:t1,state:y1.to_vec(),value:g1});}
            else{for _ in 0..self.options.max_root_iterations{
                let theta=lo+(hi-lo)*0.5;let t=t0+(t1-t0)*theta;
                let state=hermite(theta,t1-t0,y0,f0,y1,f1)?;let g=self.evaluate(t,&state)?[i];
                let tol=self.options.absolute_time_tolerance+self.options.relative_time_tolerance*t.abs();
                if g==0.0||(hi-lo)*(t1-t0).abs()<=tol||theta==lo||theta==hi{
                    hit=Some(EventOccurrence{index:i,time:t,state,value:g});break;
                }
                if g.signum()==gl.signum(){lo=theta;gl=g;}else{hi=theta;}
            }}
            let hit=hit.ok_or(IntegrationError::Callback("event root iteration budget exhausted"))?;
            let duplicate=self.last[i].is_some_and(|last|(hit.time-last).abs()<=self.options.absolute_time_tolerance+self.options.relative_time_tolerance*hit.time.abs());
            if !duplicate{hits.push(hit);}
        }
        let sign=(t1-t0).signum();hits.sort_by(|a,b|(sign*a.time).total_cmp(&(sign*b.time)).then(a.index.cmp(&b.index)));
        let terminal_time=hits.iter().find(|h|self.specs[h.index].terminal).map(|h|h.time);
        let mut stop=None;
        for hit in hits{
            if terminal_time.is_some_and(|t|sign*(hit.time-t)>0.0){break;}
            if self.specs[hit.index].terminal&&stop.is_none(){stop=Some(OdeSample{time:hit.time,state:hit.state.clone()});}
            self.push(hit)?;
        }
        self.previous=current;Ok(stop)
    }
}
fn hermite(s:f64,h:f64,y0:&[f64],f0:&[f64],y1:&[f64],f1:&[f64])->Result<Vec<f64>,IntegrationError>{
    let s2=s*s;let s3=s2*s;let mut y=Vec::with_capacity(y0.len());
    for i in 0..y0.len(){y.push(crate::error::checked((2.0*s3-3.0*s2+1.0)*y0[i]+(s3-2.0*s2+s)*h*f0[i]
        +(-2.0*s3+3.0*s2)*y1[i]+(s3-s2)*h*f1[i],"event interpolation")?);}Ok(y)
}
fn initial_report(t:f64,y:&[f64],save:bool)->OdeReport{OdeReport{time:t,state:y.to_vec(),accepted_steps:0,rejected_steps:0,evaluations:0,
    status:OdeStatus::Event,samples:if save{vec![OdeSample{time:t,state:y.to_vec()}]}else{Vec::new()}}}
fn validate_start(t0:f64,t1:f64,y:&[f64])->Result<(),IntegrationError>{if !t0.is_finite()||!t1.is_finite()||!(t1-t0).is_finite()||y.is_empty()||y.iter().any(|v|!v.is_finite()){
    Err(IntegrationError::NonFiniteInput("event initial time/state"))}else{Ok(())}}
/// Direction means increasing/decreasing in PHYSICAL time, including backward solves.
/// Initial exact roots are reported irrespective of direction; terminal ones stop.
pub fn solve_ivp_events(f:impl FnMut(f64,&[f64],&mut[f64])->Result<(),IntegrationError>,t0:f64,t1:f64,y:&[f64],
options:Rk45Options,event:impl FnMut(f64,&[f64],&mut[f64])->Result<(),IntegrationError>,specs:&[EventSpec],event_options:EventOptions)->Result<EventReport,IntegrationError>{
    validate_start(t0,t1,y)?;
    crate::solve_ivp(|_,_,_|Ok(()),t0,t0,y,options)?;
    let mut tracker=Tracker::new(event,specs,event_options,t0,y)?;
    let ode=if tracker.initial(t0,y)?{initial_report(t0,y,options.save_trajectory)}else{
        crate::ode::solve_ivp_observed(f,t0,t1,y,options,|a,b,c,d,e,f|tracker.step(a,b,c,d,e,f))?};
    Ok(EventReport{ode,events:tracker.hits,event_evaluations:tracker.eval})
}
pub fn solve_bdf1_events(f:impl FnMut(f64,&[f64],&mut[f64])->Result<(),IntegrationError>,jac:Option<&mut Jacobian<'_>>,t0:f64,t1:f64,y:&[f64],
options:Bdf1Options,event:impl FnMut(f64,&[f64],&mut[f64])->Result<(),IntegrationError>,specs:&[EventSpec],event_options:EventOptions)->Result<StiffEventReport,IntegrationError>{
    validate_start(t0,t1,y)?;
    crate::solve_bdf1(|_,_,_|Ok(()),None,t0,t0,y,options)?;
    let mut tracker=Tracker::new(event,specs,event_options,t0,y)?;
    let stiff=if tracker.initial(t0,y)?{StiffReport{ode:initial_report(t0,y,options.integration.save_trajectory),jacobian_evaluations:0,factorizations:0,newton_iterations:0}}
        else{crate::stiff::solve_bdf1_observed(f,jac,t0,t1,y,options,|a,b,c,d,e,f|tracker.step(a,b,c,d,e,f))?};
    Ok(StiffEventReport{stiff,events:tracker.hits,event_evaluations:tracker.eval})
}
