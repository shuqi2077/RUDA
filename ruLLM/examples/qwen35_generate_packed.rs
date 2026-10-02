//! Explicit packed AWQ text inference. No full floating weight expansion.
//! Run with an AWQ GEMM 4-bit checkpoint; prints packed/dense projection counts.
use half::bf16;
use ruda_driver_cuda::{CudaRuntime, CudaDevice};
use rullm::{
    CausalModelLimits, GreedyGenerationConfig, generate_causal_greedy, load_huggingface_qwen35_text_packed,
};
use std::{error::Error, io, path::Path, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    if std::env::var_os("RUDA_CUDA_COMPILER").is_none() {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(std::env::args_os().skip(1))
            .env("RUDA_CUDA_COMPILER", "ptx")
            .status()?;
        std::process::exit(status.code().unwrap_or(1));
    }
    let mut args = std::env::args().skip(1);
    let directory = args.next().ok_or_else(|| {
        io::Error::other(
            "usage: qwen35_generate_packed <model-directory> <text-prompt> [max-new-tokens] [runs]",
        )
    })?;
    let prompt = args
        .next()
        .ok_or_else(|| io::Error::other("text prompt required"))?;
    let max_new_tokens = args.next().map(|s| s.parse()).transpose()?.unwrap_or(8);
    let runs = args
        .next()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    if runs == 0 || args.next().is_some() {
        return Err(io::Error::other("invalid arguments").into());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(Path::new(&directory).join("tokenizer.json"))
        .map_err(io::Error::other)?;
    let ids = tokenizer
        .encode(prompt, false)
        .map_err(io::Error::other)?
        .get_ids()
        .iter()
        .map(|&id| i32::try_from(id))
        .collect::<Result<Vec<_>, _>>()?;
    let device = CudaDevice::default();
    let start = Instant::now();
    let loaded = load_huggingface_qwen35_text_packed::<CudaRuntime, bf16, i32, u8>(&directory, &device)?;
    if loaded.packed_awq_linears == 0 || loaded.dequantized_awq_linears != 0 {
        return Err(io::Error::other("expected packed AWQ projections and no dense AWQ expansion").into());
    }
    println!(
        "{}",
        serde_json::json!({"stage":"loaded","seconds":start.elapsed().as_secs_f64(),
        "applied_tensors":loaded.report.applied_tensors,"unloaded_tensors":loaded.unloaded_tensors,
        "prompt_token_ids":ids,"packed_awq_linears":loaded.packed_awq_linears,
        "dequantized_awq_linears":loaded.dequantized_awq_linears})
    );
    let limits = CausalModelLimits {
        vocab_size: loaded.model.config.vocab_size,
        max_sequence_length: loaded.model.config.max_position_embeddings,
    };
    for run in 1..=runs {
        let start = Instant::now();
        let output = generate_causal_greedy(
            &loaded.model,
            &limits,
            &ids,
            &GreedyGenerationConfig {
                max_new_tokens,
                eos_token_ids: vec![loaded.model.config.eos_token_id],
            },
            &device,
        )?;
        let tokens = output
            .generated_token_ids
            .iter()
            .map(|&n| n as u32)
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::json!({"stage":"generated","run":run,
            "seconds":start.elapsed().as_secs_f64(),"generated_token_ids":tokens,
            "text":tokenizer.decode(&tokens,true).map_err(io::Error::other)?,"stopped_on_eos":output.stopped_on_eos})
        );
    }
    Ok(())
}
