// SPDX-License-Identifier: Apache-2.0
//! Measure guarded fused AdamW against an explicit materialized-gradient baseline.
//! Includes the SAME device norm + compact host readback in both paths.
#[path = "../tests/fused_adamw/common.rs"] mod common;
#[path = "gradient_guard/staged.rs"] mod staged;
use common::*;
use ruda_driver_cuda::CudaRuntime;
use ruda_core::tensor::DType;
use ruda_optim::fused_adamw::{
    AdamWEntry, AdamWOptions, AdamWState, StepControl, guarded_adamw_step,
    gradient_norm::GradientGuardOptions,
};
use serde_json::json;
use std::{error::Error, path::PathBuf, time::Instant};

type State = Option<AdamWState<CudaRuntime>>;
struct Args { n: usize, tensors: usize, iterations: usize, samples: usize, warmup: usize, dtype: DType, amsgrad: bool, out: Option<PathBuf> }
fn parse() -> Result<Args, Box<dyn Error>> {
    let mut a = Args { n: 65536, tensors: 4, iterations: 5, samples: 5, warmup: 2, dtype: DType::BF16, amsgrad: false, out: None };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--elements" => a.n = args.next().ok_or("missing elements")?.parse()?,
            "--tensors" => a.tensors = args.next().ok_or("missing tensors")?.parse()?,
            "--iterations" => a.iterations = args.next().ok_or("missing iterations")?.parse()?,
            "--samples" => a.samples = args.next().ok_or("missing samples")?.parse()?,
            "--warmup" => a.warmup = args.next().ok_or("missing warmup")?.parse()?,
            "--amsgrad" => a.amsgrad = true,
            "--dtype" => a.dtype = match args.next().ok_or("missing dtype")?.as_str() {
                "f32" => DType::F32, "f16" => DType::F16, "bf16" => DType::BF16,
                _ => return Err("dtype must be f32, f16 or bf16".into()),
            },
            "--out" => a.out = Some(args.next().ok_or("missing path")?.into()),
            "--help" => { println!("gradient-guard-bench --elements N --tensors N --iterations N --samples N --warmup N --dtype f32|f16|bf16 [--amsgrad] [--out file.json]"); std::process::exit(0); }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if a.n == 0 || a.n > u32::MAX as usize / 4 || a.tensors == 0 || a.tensors > 1024
        || a.iterations == 0 || a.samples < 3 || a.warmup == 0
    { return Err("invalid size/count; require 1..1024 tensors, positive iterations/warmup, samples >=3".into()); }
    a.n.checked_mul(a.tensors).and_then(|n| n.checked_mul(64)).ok_or("memory estimate overflow")?;
    if a.out.as_ref().is_some_and(|p| p.exists()) { return Err("refusing to overwrite report".into()); }
    Ok(a)
}
fn steps(mut p: Vec<Tensor>, mut s: Vec<State>, g: &[Tensor], o: &AdamWOptions, count: usize, fused: bool) -> (Vec<Tensor>, Vec<State>) {
    for _ in 0..count {
        let entries: Vec<_> = p.iter().zip(g).zip(&s).map(|((p, g), s)| AdamWEntry { parameters: p, gradients: g, state: s.as_ref() }).collect();
        let control = StepControl { gradient_scale: 128.0, skip_update: false };
        let result = if fused { guarded_adamw_step(&entries, o, control, GradientGuardOptions::default()) }
            else { staged::step(&entries, o, control, GradientGuardOptions::default()) }.expect("guard/update failed");
        assert!(result.skip_reason.is_none(), "benchmark must execute updates");
        (p, s) = result.updates.into_iter().map(|u| (u.parameters, u.state)).unzip();
    }
    (p, s)
}
fn check(a: &(Vec<Tensor>, Vec<State>), b: &(Vec<Tensor>, Vec<State>)) {
    assert_eq!(a.0.len(), b.0.len());
    for i in 0..a.0.len() {
        close(&floats(&a.0[i]), &floats(&b.0[i]), 1e-5);
        let x = a.1[i].as_ref().unwrap(); let y = b.1[i].as_ref().unwrap();
        assert_eq!(x.step(), y.step());
        close_moments(&floats(x.first_moment()), &floats(y.first_moment()));
        close_moments(&floats(x.second_moment()), &floats(y.second_moment()));
        match (x.max_second_moment(), y.max_second_moment()) {
            (Some(x), Some(y)) => close_moments(&floats(x), &floats(y)),
            (None, None) => (), _ => panic!("maximum state mismatch"),
        }
    }
}
fn measure(p: &[Tensor], s: &[State], g: &[Tensor], o: &AdamWOptions, count: usize, fused: bool) -> (f64, (Vec<Tensor>, Vec<State>)) {
    sync(); let now = Instant::now();
    let result = steps(p.to_vec(), s.to_vec(), g, o, count, fused);
    sync();
    (now.elapsed().as_secs_f64() * 1e3 / count as f64, result)
}
fn distribution(samples: &[f64]) -> serde_json::Value {
    let mut sorted = samples.to_vec(); sorted.sort_by(f64::total_cmp);
    let m = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 { (sorted[m-1] + sorted[m]) * 0.5 } else { sorted[m] };
    json!({"median_ms":median, "min_ms":sorted[0], "max_ms":sorted[sorted.len()-1], "samples_ms":samples})
}
fn main() -> Result<(), Box<dyn Error>> {
    let a = parse()?;
    let compiler = std::env::var("RUDA_CUDA_COMPILER").unwrap_or_else(|_| "nvrtc".to_owned());
    if compiler != "nvrtc" && compiler != "ptx" { return Err("select nvrtc or ptx explicitly".into()); }
    eprintln!("elements_per_tensor={} tensors={} rough_payload_budget_hint={} bytes (not a bound or allocator/device-memory measurement)", a.n, a.tensors, a.n*a.tensors*64);
    let p: Vec<_> = (0..a.tensors).map(|t| tensor(&(0..a.n).map(|i| ((i+t)%97) as f32*0.01-0.5).collect::<Vec<_>>(), [a.n], DType::F32)).collect();
    let g: Vec<_> = (0..a.tensors).map(|t| tensor(&(0..a.n).map(|i| (((i+t)%31) as f32-15.0)*0.125*128.0).collect::<Vec<_>>(), [a.n], a.dtype)).collect();
    let o = AdamWOptions { amsgrad: a.amsgrad, ..Default::default() };
    // Establish identical, materialized nonzero state before warmup/timed runs.
    let initial = steps(p, (0..a.tensors).map(|_| None).collect(), &g, &o, 1, true);
    sync();
    let warm_f = steps(initial.0.clone(), initial.1.clone(), &g, &o, a.warmup, true);
    let warm_b = steps(initial.0.clone(), initial.1.clone(), &g, &o, a.warmup, false);
    sync(); check(&warm_f, &warm_b); drop((warm_f, warm_b));
    let (mut fused, mut baseline) = (Vec::new(), Vec::new());
    for sample in 0..a.samples {
        let (f, b) = if sample % 2 == 0 {
            (measure(&initial.0,&initial.1,&g,&o,a.iterations,true), measure(&initial.0,&initial.1,&g,&o,a.iterations,false))
        } else {
            let b = measure(&initial.0,&initial.1,&g,&o,a.iterations,false);
            (measure(&initial.0,&initial.1,&g,&o,a.iterations,true), b)
        };
        check(&f.1, &b.1); // readback and comparisons excluded from measured interval
        fused.push(f.0); baseline.push(b.0);
        eprintln!("sample={} fused_ms={} materialized_ms={} correctness=passed", sample+1, f.0, b.0);
    }
    let f = distribution(&fused); let b = distribution(&baseline);
    let report = json!({
        "schema":"ruda.gradient_guard.benchmark.v1", "status":"passed", "compiler":compiler,
        "runtime_info":format!("{:?}", initial.0[0].client.info()), "device":format!("{:?}", initial.0[0].device),
        "elements_per_tensor":a.n, "tensors":a.tensors, "gradient_dtype":format!("{:?}", a.dtype),
        "iterations":a.iterations, "warmup":a.warmup, "amsgrad":a.amsgrad,
        "timing_scope":"synchronized wall time, including norm kernels, host readback/decision, allocations, submission and final completion; not GPU-only time",
        "correctness":"parameters and all moments checked after warmup and each measured batch; run gradient-guard-cuda first for independent CPU oracle",
        "fused":f, "materialized_baseline":b,
        "measured_speedup":b["median_ms"].as_f64().unwrap()/f["median_ms"].as_f64().unwrap(),
        "source_level_savings_per_step":{"clip_launches":a.tensors,"temporary_gradient_bytes":4*a.n*a.tensors,"logical_gradient_bytes":8*a.n*a.tensors},
        "comparison_scope":"same scaled-L2 implementation and host synchronization in both; not a comparison to PyTorch fused/foreach or device-only GradScaler",
        "no_device_only_capture_claim":true
    });
    let text = serde_json::to_string_pretty(&report)?;
    if let Some(path) = a.out {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(text.as_bytes())?; file.write_all(b"\n")?;
    }
    println!("{text}"); Ok(())
}
