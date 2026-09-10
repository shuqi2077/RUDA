// SPDX-License-Identifier: Apache-2.0
//! Synchronized CUDA A/B experiment, not a pre-recorded speed claim.
#[path = "fused_adamw/staged.rs"]
mod staged;
#[path = "../tests/fused_adamw/common.rs"]
mod common;
use common::*;
use ruda_core::tensor::DType;
use ruda_driver_cuda::CudaRuntime;
use ruda_optim::fused_adamw::{AdamWOptions, AdamWState, StepControl, adamw_step, reference::adamw_reference};
use serde_json::json;
use std::{error::Error, path::PathBuf, time::Instant};

type State = AdamWState<CudaRuntime>;

struct Args { n: usize, iterations: usize, samples: usize, warmup: usize, dtype: DType, amsgrad: bool, out: Option<PathBuf> }
fn parse() -> Result<Args, Box<dyn Error>> {
    let mut result = Args { n: 65536, iterations: 20, samples: 7, warmup: 3, dtype: DType::F32, amsgrad: false, out: None };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--elements" => result.n = args.next().ok_or("missing elements")?.parse()?,
            "--iterations" => result.iterations = args.next().ok_or("missing iterations")?.parse()?,
            "--samples" => result.samples = args.next().ok_or("missing samples")?.parse()?,
            "--warmup" => result.warmup = args.next().ok_or("missing warmup")?.parse()?,
            "--dtype" => result.dtype = match args.next().ok_or("missing dtype")?.as_str() {
                "f32" => DType::F32, "f16" => DType::F16, "bf16" => DType::BF16,
                _ => return Err("dtype must be f32, f16 or bf16".into()),
            },
            "--amsgrad" => result.amsgrad = true,
            "--out" => result.out = Some(args.next().ok_or("missing output path")?.into()),
            "--help" => {
                println!("fused-adamw-bench [--elements N] [--iterations N] [--samples N] [--warmup N] [--dtype f32|f16|bf16] [--amsgrad] [--out result.json]");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if result.n == 0 || result.n > u32::MAX as usize / 4 || result.iterations == 0
        || result.samples < 3 || result.warmup == 0
    { return Err("require 1..u32::MAX/4 elements, positive iterations/warmup, and at least 3 samples".into()); }
    if result.out.as_ref().is_some_and(|p| p.exists()) {
        return Err("refusing to overwrite existing benchmark report".into());
    }
    Ok(result)
}

fn run_steps(mut p: Tensor, mut s: Option<State>, g: &Tensor, o: &AdamWOptions, count: usize, fused: bool) -> (Tensor, Option<State>) {
    for _ in 0..count {
        let result = if fused { adamw_step(&p, g, s.as_ref(), o, StepControl::default()).unwrap() }
            else { staged::staged_step(&p, g, s.as_ref(), o) };
        p = result.parameters;
        s = result.state;
    }
    (p, s)
}
fn measure(p: &Tensor, s: &Option<State>, g: &Tensor, o: &AdamWOptions, count: usize, fused: bool) -> (f64, Tensor, Option<State>) {
    sync();
    let start = Instant::now();
    let (out, state) = run_steps(p.clone(), s.clone(), g, o, count, fused);
    sync(); // Includes allocations, host submission, device work and completion.
    (start.elapsed().as_secs_f64() * 1e3 / count as f64, out, state)
}
fn stats(samples: &[f64]) -> serde_json::Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 { (sorted[mid - 1] + sorted[mid]) * 0.5 } else { sorted[mid] };
    json!({ "median_ms": median, "min_ms": sorted[0], "max_ms": sorted[sorted.len()-1], "samples_ms": samples })
}
fn check(a: &(f64, Tensor, Option<State>), b: &(f64, Tensor, Option<State>)) {
    close(&floats(&a.1), &floats(&b.1), 1e-5);
    let x = a.2.as_ref().unwrap(); let y = b.2.as_ref().unwrap();
    assert_eq!(x.step(), y.step());
    close_moments(&floats(x.first_moment()), &floats(y.first_moment()));
    close_moments(&floats(x.second_moment()), &floats(y.second_moment()));
    if let (Some(x), Some(y)) = (x.max_second_moment(), y.max_second_moment()) {
        close_moments(&floats(x), &floats(y));
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse()?;
    let backend = std::env::var("RUDA_CUDA_COMPILER").unwrap_or_else(|_| "nvrtc".to_owned());
    if backend != "nvrtc" && backend != "ptx" { return Err("select nvrtc or ptx explicitly".into()); }
    let p_values: Vec<_> = (0..args.n).map(|i| ((i % 101) as f32 - 50.0) * 0.01).collect();
    let g_values = rounded(&(0..args.n).map(|i| ((i % 37) as f32 - 18.0) * 0.001).collect::<Vec<_>>(), args.dtype);
    let p = tensor(&p_values, [args.n], DType::F32);
    let g = tensor(&g_values, [args.n], args.dtype);
    let o = AdamWOptions { amsgrad: args.amsgrad, weight_decay: 0.01, ..Default::default() };
    let expected = adamw_reference(&p_values, &g_values, None, &o, StepControl::default())?;
    let first = adamw_step(&p, &g, None, &o, StepControl::default())?;
    close(&floats(&first.parameters), &expected.parameters, 1e-5);
    check_state(first.state.as_ref().unwrap(), expected.state.as_ref().unwrap());
    // Both paths start from exactly the same materialized nonzero optimizer state.
    let p = first.parameters;
    let s = first.state;
    let warm_f = run_steps(p.clone(), s.clone(), &g, &o, args.warmup, true);
    let warm_b = run_steps(p.clone(), s.clone(), &g, &o, args.warmup, false);
    sync();
    close(&floats(&warm_f.0), &floats(&warm_b.0), 1e-5);
    drop((warm_f, warm_b));
    let (mut fused_times, mut staged_times) = (Vec::new(), Vec::new());
    for sample in 0..args.samples {
        let (fused, baseline) = if sample % 2 == 0 {
            (measure(&p, &s, &g, &o, args.iterations, true), measure(&p, &s, &g, &o, args.iterations, false))
        } else {
            let b = measure(&p, &s, &g, &o, args.iterations, false);
            let f = measure(&p, &s, &g, &o, args.iterations, true);
            (f, b)
        };
        check(&fused, &baseline); // Readback and correctness checks are outside timing.
        fused_times.push(fused.0); staged_times.push(baseline.0);
        eprintln!("sample={} fused_ms={} staged_ms={} correctness=passed", sample + 1, fused.0, baseline.0);
    }
    let f = stats(&fused_times); let b = stats(&staged_times);
    let report = json!({
        "schema": "ruda.fused_adamw.benchmark.v1", "status": "passed", "compiler": backend,
        "device": format!("{:?}", p.device), "runtime_info": format!("{:?}", p.client.info()),
        "elements": args.n, "gradient_dtype": format!("{:?}", args.dtype), "master_dtype": "F32",
        "amsgrad": args.amsgrad, "iterations_per_sample": args.iterations, "warmup_steps": args.warmup,
        "timing_scope": "synchronized wall time per step; includes allocation, host submission, execution and final sync; not GPU-only time",
        "correctness": "first step vs FP64 CPU oracle; every measured batch vs staged CUDA parameters and moments",
        "fused": f, "staged_baseline": b,
        "measured_speedup": b["median_ms"].as_f64().unwrap() / f["median_ms"].as_f64().unwrap(),
        "planned_launches_per_step": {"fused": 1, "staged": if args.amsgrad {5} else {4}},
        "logical_bytes_per_parameter_steady_state": {
            "fused": 24 + args.dtype.size() + if args.amsgrad {8} else {0},
            "staged": 40 + 2 * args.dtype.size() + if args.amsgrad {12} else {0},
            "scope": "source-level traffic accounting, not measured DRAM bytes; excludes allocator and cache effects"
        },
        "comparison_scope": "explicitly staged reference kernels; NOT PyTorch fused AdamW and NOT the existing Ruda fusion backend"
    });
    let text = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.out {
        // create_new avoids an accidental overwrite if another benchmark wins a race.
        use std::io::Write;
        let mut output = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
        output.write_all(text.as_bytes())?;
        output.write_all(b"\n")?;
    }
    println!("{text}");
    Ok(())
}
