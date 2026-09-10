// SPDX-License-Identifier: Apache-2.0
//! Small direct-solver A/B benchmark. Measures synchronized API wall time,
//! INCLUDING allocation, submission and completion (not GPU event time).
#[path="../tests/warp_common/mod.rs"]mod common;
use common::*;
use rusolver::kernel_plan::{WarpDirectKind,WarpDirectPlan};
use serde_json::json;
use std::{error::Error,path::PathBuf,time::Instant,io::Write};
struct Args{batch:usize,n:usize,nr:usize,warmup:usize,iterations:usize,samples:usize,kind:Kind,out:Option<PathBuf>}
fn parse()->Result<Args,Box<dyn Error>>{
    let mut a=Args{batch:256,n:16,nr:4,warmup:2,iterations:3,samples:7,kind:Kind::Cholesky,out:None};
    let mut args=std::env::args().skip(1);
    while let Some(flag)=args.next(){
        if flag=="--help"{println!("--kind cholesky|lu --batch N --order 1..32 --rhs 1..8 --samples N --iterations N --warmup N [--out file.json]");std::process::exit(0);}
        let value=args.next().ok_or("missing argument value")?;
        match flag.as_str(){
            "--batch"=>a.batch=value.parse()?,"--order"=>a.n=value.parse()?,"--rhs"=>a.nr=value.parse()?,
            "--warmup"=>a.warmup=value.parse()?,"--iterations"=>a.iterations=value.parse()?,"--samples"=>a.samples=value.parse()?,
            "--kind"=>a.kind=match value.as_str(){"lu"=>Kind::Lu,"cholesky"=>Kind::Cholesky,_=>return Err("unknown solver".into())},
            "--out"=>a.out=Some(value.into()),_=>return Err(format!("unknown flag {flag}").into()),
        }
    }
    WarpDirectPlan::new(a.batch,a.n,a.nr,match a.kind{Kind::Lu=>WarpDirectKind::Lu,Kind::Cholesky=>WarpDirectKind::Cholesky})?;
    if a.batch==0||a.batch>65536||a.warmup==0||a.warmup>100||a.samples<3||a.samples>100||a.iterations==0||a.iterations>100{
        return Err("bounded benchmark requires 1..65536 systems, warmup/iterations 1..100 and samples 3..100".into());
    }
    if a.out.as_ref().is_some_and(|p|p.exists()){return Err("refusing to overwrite benchmark evidence".into());}
    Ok(a)
}
fn dist(values:&[f64])->serde_json::Value{
    let mut sorted=values.to_vec();sorted.sort_by(f64::total_cmp);let m=sorted.len()/2;
    let median=if sorted.len()%2==0{(sorted[m-1]+sorted[m])*0.5}else{sorted[m]};
    json!({"median_ms":median,"min_ms":sorted[0],"max_ms":sorted[sorted.len()-1],"samples_ms":values})
}
fn measure(c:&Case,a:&Tensor,b:&Tensor,count:usize,warp:bool)->(f64,Vec<f64>){
    let mut times=Vec::new();
    for _ in 0..count{
        sync();let start=Instant::now();let output=run(c.kind,warp,a,b);sync();
        let ms=start.elapsed().as_secs_f64()*1000.;
        assert!(ms.is_finite()&&ms>0.0,"invalid timer sample");
        // Check EVERY timed output, but exclude readback/reference checks from time.
        c.check(&output);times.push(ms);
    }
    (times.iter().sum::<f64>()/times.len()as f64,times)
}
fn main()->Result<(),Box<dyn Error>>{
    let args=parse()?;let compiler=std::env::var("RUDA_CUDA_COMPILER").unwrap_or_else(|_|"nvrtc".into());
    if compiler!="nvrtc"&&compiler!="ptx"{return Err("RUDA_CUDA_COMPILER must be nvrtc or ptx".into());}
    let kind=match args.kind{Kind::Lu=>WarpDirectKind::Lu,Kind::Cholesky=>WarpDirectKind::Cholesky};
    let plan=WarpDirectPlan::new(args.batch,args.n,args.nr,kind)?;
    eprintln!("batch={} n={} nrhs={} shared_bytes/block={} (not measured PPA)",args.batch,args.n,args.nr,plan.shared_bytes());
    let case=Case::new(args.kind,args.batch,args.n,args.nr,1.0);let(a,b)=case.upload();
    for _ in 0..args.warmup{let serial=run(case.kind,false,&a,&b);let warp=run(case.kind,true,&a,&b);case.check(&serial);case.check(&warp);compare(&serial,&warp);}
    let(mut serial_samples,mut warp_samples,mut raw)=(Vec::new(),Vec::new(),Vec::new());
    for index in 0..args.samples{
        let(s,w)=if index%2==0{(measure(&case,&a,&b,args.iterations,false),measure(&case,&a,&b,args.iterations,true))}else{
            let w=measure(&case,&a,&b,args.iterations,true);(measure(&case,&a,&b,args.iterations,false),w)
        };
        serial_samples.push(s.0);warp_samples.push(w.0);raw.push(json!({"serial_ms":s.1,"warp_ms":w.1}));
        eprintln!("sample={} serial_ms={} warp_ms={} numerical_check=passed",index+1,s.0,w.0);
    }
    bits_eq(&floats(&a),&case.av);bits_eq(&floats(&b),&case.bv);
    let serial=dist(&serial_samples);let warp=dist(&warp_samples);
    let ratio=serial["median_ms"].as_f64().unwrap()/warp["median_ms"].as_f64().unwrap();
    let report=json!({"schema":"ruda.warp_direct.benchmark.v1","status":"passed","kind":format!("{:?}",kind),
        "batch":args.batch,"order":args.n,"rhs":args.nr,"dtype":"FP32","compiler":compiler,
        "device":format!("{:?}",a.device),"runtime_info":format!("{:?}",a.client.info()),"hardware":format!("{:?}",a.client.properties().hardware),
        "warmup":args.warmup,"iterations_per_sample":args.iterations,"serial":serial,"warp_shared":warp,"raw_iterations":raw,
        "measured_api_speedup":ratio,"timing_scope":"synchronized wall time per API call: includes output allocation, launch/submission and completion; excludes JIT warmup, uploads, result readback and correctness checks; not GPU-only event time",
        "correctness":"every timed output checked vs FP64 host solve plus independent FP64 residual and factor reconstruction; inputs preserved",
        "algorithm_scope":"n<=32, nrhs<=8, one warp/block/system; original serial kernels retained; not cuSOLVER comparison",
        "source_metrics":{"shared_bytes_per_block":plan.shared_bytes(),"pitch":plan.pitch(),"logical_global_bytes_per_system":plan.logical_global_bytes_per_system()},
        "source_metrics_are_not_memory_controller_counters":true,"source_snapshot":include_str!("../src/tensor/warp_kernel.rs")});
    let text=serde_json::to_string_pretty(&report)?;
    if let Some(path)=args.out{let mut f=std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;f.write_all(text.as_bytes())?;f.write_all(b"\n")?;}
    println!("{text}");Ok(())
}
