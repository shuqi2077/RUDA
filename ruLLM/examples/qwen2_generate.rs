use half::bf16;
use ruda_tensor::api::backend::Backend;
use ruda_tensor_device::cuda::{Cuda, CudaDevice};
use rullm::{
    GreedyGenerationConfig, generate_greedy_packed_ruda, load_huggingface_qwen2_pipeline,
};
use std::{error::Error, io, time::Instant};

type ModelBackend = Cuda<bf16, i32>;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let directory = args.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: qwen2_generate <model-directory> <prompt> [max-new-tokens] [runs]",
        )
    })?;
    let prompt = args.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "a prompt is required")
    })?;
    let max_new_tokens = args.next().map(|s| s.parse()).transpose()?.unwrap_or(32);
    let runs = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(1);
    if runs == 0 || args.next().is_some() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid run arguments").into());
    }
    let device = CudaDevice::default();
    let started = Instant::now();
    let pipeline = load_huggingface_qwen2_pipeline::<ModelBackend>(&directory, &device)?;
    let prompt_ids = pipeline.encode(&prompt, false)?;
    let config = pipeline.loaded.config;
    let model = pipeline.loaded.model.into_packed_inference();
    ModelBackend::sync(&device)?;
    println!("{}", serde_json::json!({
        "stage": "loaded", "backend": "ruda-nvidia", "dtype": "bf16",
        "seconds": started.elapsed().as_secs_f64(), "prompt_token_ids": prompt_ids,
        "applied_tensors": pipeline.loaded.report.applied_tensors,
    }));
    for run in 1..=runs {
        let started = Instant::now();
        let output = generate_greedy_packed_ruda(
            &model, &config, &prompt_ids,
            &GreedyGenerationConfig {
                max_new_tokens,
                eos_token_ids: pipeline.loaded.default_eos_token_ids.clone(),
            },
            &device,
        )?;
        let seconds = started.elapsed().as_secs_f64();
        let ids = output.generated_token_ids.iter().map(|&id| u32::try_from(id))
            .collect::<Result<Vec<_>, _>>()?;
        let text = pipeline.tokenizer.decode(&ids, true).map_err(io::Error::other)?;
        println!("{}", serde_json::json!({
            "stage": "generated", "run": run, "seconds": seconds,
            "tokens_per_second": ids.len() as f64 / seconds,
            "generated_token_ids": output.generated_token_ids,
            "stopped_on_eos": output.stopped_on_eos, "text": text,
        }));
    }
    Ok(())
}
