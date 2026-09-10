// SPDX-License-Identifier: Apache-2.0
//! Adaptive backward Euler (BDF1), Newton solves, optional analytic Jacobian.
//! Step doubling uses TWO half steps as the accepted solution (no extrapolation,
//! preserving backward Euler damping). First-order and expensive but L-stable.
use crate::{IntegrationError,Rk45Options,OdeReport,OdeSample,OdeStatus};
use crate::error::{checked,validate_tolerance,vector};
use rusolver::{Lu,Matrix,Tolerance};
pub type Jacobian<'a> = dyn FnMut(f64,&[f64],&mut[f64])->Result<(),IntegrationError>+'a;
#[derive(Clone,Copy,Debug)]pub struct Bdf1Options {
    pub integration:Rk45Options,pub max_newton_iterations:usize,
    pub max_backtracks:usize,pub newton_tolerance:f64,pub max_dimension:usize,
}
impl Default for Bdf1Options{fn default()->Self{Self{integration:Rk45Options{absolute_tolerance:1e-7,relative_tolerance:1e-5,..Default::default()},
    max_newton_iterations:12,max_backtracks:8,newton_tolerance:0.03,max_dimension:256}}}
#[derive(Clone,Debug)]pub struct StiffReport{pub ode:OdeReport,pub jacobian_evaluations:usize,pub factorizations:usize,pub newton_iterations:usize}
struct Counters{eval:usize,jac:usize,lu:usize,newton:usize}
fn eval(f:&mut impl FnMut(f64,&[f64],&mut[f64])->Result<(),IntegrationError>,t:f64,y:&[f64],c:&mut Counters,max:usize)->Result<Option<Vec<f64>>,IntegrationError>{
    if c.eval>=max{return Ok(None);}let mut dy=vector(y.len())?;dy.fill(f64::NAN);f(t,y,&mut dy)?;c.eval+=1;
    for(i,v)in dy.iter().enumerate(){if !v.is_finite(){return Err(IntegrationError::NonFiniteFunction{at:t,index:i});}}Ok(Some(dy))
}
fn scaled_norm(x:&[f64],a:&[f64],b:&[f64],o:Rk45Options)->f64{x.iter().zip(a).zip(b).fold(0.0f64,|n,((&x,&a),&b)|{
    let scale=o.absolute_tolerance+o.relative_tolerance*a.abs().max(b.abs());
    n.max(if scale==0.0{if x==0.0{0.0}else{f64::INFINITY}}else{x.abs()/scale})})}
fn residual(y:&[f64],old:&[f64],f:&[f64],h:f64)->Result<Vec<f64>,IntegrationError>{
    y.iter().zip(old).zip(f).map(|((&y,&o),&f)|checked((y-o)-h*f,"implicit residual")).collect()
}
fn implicit_step(f:&mut impl FnMut(f64,&[f64],&mut[f64])->Result<(),IntegrationError>,jac:&mut Option<&mut Jacobian<'_>>,
t:f64,old:&[f64],h:f64,o:Bdf1Options,c:&mut Counters)->Result<Option<(Vec<f64>,Vec<f64>)>,IntegrationError>{
    let n=old.len();let mut y=old.to_vec();let max=o.integration.max_evaluations;
    for _ in 0..o.max_newton_iterations{
        let Some(fy)=eval(f,t,&y,c,max)?else{return Ok(None)};
        let r=residual(&y,old,&fy,h)?;let rn=scaled_norm(&r,old,&y,o.integration);
        if rn<=o.newton_tolerance{return Ok(Some((y,fy)));}
        c.newton+=1;let mut j=vector(n.checked_mul(n).ok_or(IntegrationError::Allocation)?)?;
        c.jac+=1;
        if let Some(jacobian)=jac.as_deref_mut(){j.fill(f64::NAN);jacobian(t,&y,&mut j)?;
            if j.iter().any(|v|!v.is_finite()){return Err(IntegrationError::Callback("analytic Jacobian must fully overwrite finite n*n row-major entries"));}}
        else{for col in 0..n{
            let mut trial=y.clone();let perturb=f64::EPSILON.sqrt()*y[col].abs().max(1.0);
            trial[col]=checked(y[col]+perturb,"finite-difference Jacobian point")?;
            let actual=trial[col]-y[col];if actual==0.0{return Err(IntegrationError::Arithmetic("Jacobian perturbation underflow"));}
            let Some(ft)=eval(f,t,&trial,c,max)?else{return Ok(None)};
            for row in 0..n{j[row*n+col]=checked((ft[row]-fy[row])/actual,"finite-difference Jacobian")?;}
        }}
        for i in 0..n{for k in 0..n{j[i*n+k]=checked((if i==k{1.0}else{0.0})-h*j[i*n+k],"implicit Jacobian")?;}}
        let mat=Matrix::new(n,n,j).map_err(|_|IntegrationError::Arithmetic("implicit matrix"))?;
        c.lu+=1;
        let lu=match Lu::factor(mat.view(),Tolerance::EXACT){Ok(x)=>x,Err(_)=>return Ok(None)};
        let rhs=Matrix::new(n,1,r.iter().map(|x|-x).collect()).map_err(|_|IntegrationError::Arithmetic("Newton RHS"))?;
        let delta=match lu.solve(rhs.view()){Ok(x)=>x,Err(_)=>return Ok(None)};
        let mut step=1.0;let mut accepted=false;
        for _ in 0..=o.max_backtracks{
            let candidate:Vec<f64>=y.iter().zip(delta.values()).map(|(&y,&d)|y+step*d).collect();
            if candidate.iter().all(|x|x.is_finite()){
                let Some(fc)=eval(f,t,&candidate,c,max)?else{return Ok(None)};
                let rc=residual(&candidate,old,&fc,h)?;let norm=scaled_norm(&rc,old,&candidate,o.integration);
                if norm<=o.newton_tolerance{return Ok(Some((candidate,fc)));}
                if norm<rn{y=candidate;accepted=true;break;}
            }step*=0.5;
        }
        if !accepted{return Ok(None);}
    }Ok(None)
}
pub fn solve_bdf1(f:impl FnMut(f64,&[f64],&mut[f64])->Result<(),IntegrationError>,jacobian:Option<&mut Jacobian<'_>>,
t_start:f64,t_end:f64,y0:&[f64],options:Bdf1Options)->Result<StiffReport,IntegrationError>{
    solve_bdf1_observed(f,jacobian,t_start,t_end,y0,options,|_,_,_,_,_,_|Ok(None))
}
pub(crate)fn solve_bdf1_observed(mut f:impl FnMut(f64,&[f64],&mut[f64])->Result<(),IntegrationError>,mut jacobian:Option<&mut Jacobian<'_>>,
t_start:f64,t_end:f64,y0:&[f64],o:Bdf1Options,
mut observer:impl FnMut(f64,&[f64],&[f64],f64,&[f64],&[f64])->Result<Option<OdeSample>,IntegrationError>)->Result<StiffReport,IntegrationError>{
    let opt=o.integration;validate_tolerance(opt.absolute_tolerance,opt.relative_tolerance)?;
    if !t_start.is_finite()||!t_end.is_finite()||!(t_end-t_start).is_finite()||y0.is_empty()||y0.iter().any(|x|!x.is_finite()){
        return Err(IntegrationError::NonFiniteInput("BDF1 time/state"));}
    if y0.len()>o.max_dimension||o.max_newton_iterations==0||o.max_backtracks>64||!o.newton_tolerance.is_finite()||o.newton_tolerance<=0.0||o.newton_tolerance>=1.0
        ||!opt.min_step.is_finite()||opt.min_step<0.0||opt.max_step.is_nan()||opt.max_step<=0.0||opt.max_step<opt.min_step
        ||opt.initial_step.is_some_and(|h|!h.is_finite()||h<=0.0)||opt.max_output_points==0{
        return Err(IntegrationError::InvalidOption("BDF1 dimension/Newton/step budget"));}
    let mut report=OdeReport{time:t_start,state:y0.to_vec(),accepted_steps:0,rejected_steps:0,evaluations:0,status:OdeStatus::ReachedEnd,samples:Vec::new()};
    if opt.save_trajectory{report.samples.push(OdeSample{time:t_start,state:y0.to_vec()});}
    let mut c=Counters{eval:0,jac:0,lu:0,newton:0};
    if t_start!=t_end{
        let direction=(t_end-t_start).signum();let mut h_abs=opt.initial_step.unwrap_or((t_end-t_start).abs()*0.01).min(opt.max_step);
        let mut f_old=eval(&mut f,t_start,y0,&mut c,opt.max_evaluations)?;
        report.status=OdeStatus::MaxSteps;
        for _ in 0..opt.max_steps{
            if f_old.is_none()||c.eval>=opt.max_evaluations{report.status=OdeStatus::MaxEvaluations;break;}
            if opt.save_trajectory&&report.samples.len()>=opt.max_output_points{report.status=OdeStatus::OutputLimit;break;}
            let remaining=(t_end-report.time).abs();h_abs=h_abs.min(remaining).min(opt.max_step);
            let next=if h_abs==remaining{t_end}else{report.time+direction*h_abs};let h=next-report.time;let mid=report.time+0.5*h;
            if next==report.time||mid==report.time||mid==next||(h_abs<opt.min_step&&h_abs<remaining){report.status=OdeStatus::StepTooSmall;break;}
            let full=implicit_step(&mut f,&mut jacobian,next,&report.state,h,o,&mut c)?;
            let half=if full.is_some(){implicit_step(&mut f,&mut jacobian,mid,&report.state,mid-report.time,o,&mut c)?}else{None};
            let fine=if let Some((ref half,_))=half{implicit_step(&mut f,&mut jacobian,next,half,next-mid,o,&mut c)?}else{None};
            if c.eval>=opt.max_evaluations&&fine.is_none(){report.status=OdeStatus::MaxEvaluations;break;}
            let (Some((full,_)),Some((fine,ffine)))=(full,fine)else{report.rejected_steps+=1;h_abs*=0.25;continue};
            let err:Vec<f64>=fine.iter().zip(&full).map(|(&a,&b)|a-b).collect();let error=scaled_norm(&err,&report.state,&fine,opt);
            if error<=1.0{
                let stop=observer(report.time,&report.state,f_old.as_ref().ok_or(IntegrationError::Arithmetic("missing derivative"))?,next,&fine,&ffine)?;
                report.accepted_steps+=1;
                if let Some(stop)=stop{report.time=stop.time;report.state=stop.state;report.status=OdeStatus::Event;}
                else{report.time=next;report.state=fine;f_old=Some(ffine);}
                if opt.save_trajectory{report.samples.push(OdeSample{time:report.time,state:report.state.clone()});}
                if report.status==OdeStatus::Event{break;}
                if next==t_end{report.status=OdeStatus::ReachedEnd;break;}
                h_abs*=if error==0.0{2.0}else{(0.9/error.sqrt()).clamp(0.2,2.0)};
            }else{report.rejected_steps+=1;h_abs*=(0.9/error.sqrt()).clamp(0.1,0.8);}
        }
    }
    report.evaluations=c.eval;Ok(StiffReport{ode:report,jacobian_evaluations:c.jac,factorizations:c.lu,newton_iterations:c.newton})
}
