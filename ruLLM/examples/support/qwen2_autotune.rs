//! Offline full-stack calibration. The model is never downloaded by this example.
use ruda::runtime::tune::stack::{enable_stack_autotune, Mode, StackPolicy};
use ruda_tensor::api::backend::Backend;
use rullm::{GreedyGenerationConfig, generate_greedy_packed_with_mode, load_huggingface_qwen2_pipeline};
use rullm::autotune::calibrate_greedy_packed;
use std::{error::Error, io, path::PathBuf, time::Instant};
use super::B;

pub fn run()->Result<(),Box<dyn Error>> {
    let mut args=std::env::args().skip(1);
    let directory=args.next().ok_or_else(||io::Error::other("usage: qwen2_autotune <model-directory> <immutable-weights-revision> <prompt> [max-new-tokens] [cache-directory] [explore|cache-only|disabled]"))?;
    let revision=args.next().ok_or_else(||io::Error::other("immutable weights revision required"))?;
    let prompt=args.next().ok_or_else(||io::Error::other("prompt required"))?;
    let max_new_tokens=args.next().map(|x|x.parse()).transpose()?.unwrap_or(16);
    let cache=args.next().map(PathBuf::from).unwrap_or_else(||PathBuf::from(".ruda-stack-cache"));
    let mode=match args.next().as_deref().unwrap_or("explore") {
        "explore"=>Mode::Explore,"cache-only"=>Mode::CacheOnly,"disabled"=>Mode::Disabled,
        _=>return Err(io::Error::other("mode must be explore, cache-only or disabled").into()) };
    if args.next().is_some() {return Err(io::Error::other("too many arguments").into());}
    let tuner=enable_stack_autotune(StackPolicy {mode,..StackPolicy::default()},Some(cache))?;
    let device=Default::default();
    let pipeline=load_huggingface_qwen2_pipeline::<B>(&directory,&device)?;
    let tokens=pipeline.encode(&prompt,false)?;
    let config=pipeline.loaded.config;
    let model=pipeline.loaded.model.into_packed_inference();
    let generation=GreedyGenerationConfig {max_new_tokens,eos_token_ids:pipeline.loaded.default_eos_token_ids};
    B::sync(&device)?;
    let plan=calibrate_greedy_packed(tuner,&model,&config,&revision,&tokens,&generation,&device)?;
    println!("{}",serde_json::json!({"stage":"calibrated","mode":format!("{:?}",plan.mode),
        "source":format!("{:?}",plan.decision.source),"verified":plan.decision.verified,
        "candidate_to_reference_ratio":plan.decision.ratio,"lower_levels_unchanged":plan.lower_levels_unchanged(tuner)}));
    let start=Instant::now();
    let output=generate_greedy_packed_with_mode(&model,&config,&tokens,&generation,&device,plan.mode)?;
    B::sync(&device)?;
    println!("{}",serde_json::json!({"stage":"actual-request","seconds":start.elapsed().as_secs_f64(),
        "generated_token_ids":output.generated_token_ids,"cache_hits":tuner.stats().memory_hits,
        "disk_hits":tuner.stats().disk_hits,"calibration_passes":tuner.stats().tunes}));
    for report in tuner.reports() {
        let candidates:Vec<_>=report.candidates.iter().map(|c|serde_json::json!({"name":c.name,"verified":c.verified,
            "paired_samples":c.samples,"ratio":c.ratio,"relative_mad":c.relative_mad,"note":c.note})).collect();
        println!("{}",serde_json::json!({"stage":"tuning-report","operation":report.operation,"winner":report.winner,
            "budget_exhausted":report.budget_exhausted,"candidates":candidates}));
    }
    Ok(())
}
