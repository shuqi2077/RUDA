//! Stream token IDs as JSON lines, then decode the complete generated suffix.
use half::bf16;
use ruda_tensor_device::cuda::{Cuda, CudaDevice};
use rullm::{
    GenerationControl, SamplingConfig, SamplingGenerationConfig,
    generate_causal_sampled_stream, load_huggingface_qwen2_pipeline,
};
use std::{error::Error, io::{self, Write}, ops::ControlFlow};

type B = Cuda<bf16, i32>;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let usage = || io::Error::new(io::ErrorKind::InvalidInput,
        "usage: qwen2_stream <model-directory> <raw-prompt> [max-new-tokens] [stop-text]");
    let directory = args.next().ok_or_else(usage)?;
    let prompt = args.next().ok_or_else(usage)?;
    let max_new_tokens = args.next().map(|value| value.parse::<usize>()).transpose()?.unwrap_or(32);
    let stop_text = args.next();
    if args.next().is_some() { return Err(usage().into()); }
    let device = CudaDevice::default();
    let pipeline = load_huggingface_qwen2_pipeline::<B>(&directory, &device)?;
    let prompt_ids = pipeline.encode(&prompt, false)?;
    // A token-sequence stop is exact-ID matching, not arbitrary substring
    // matching: contextual tokenization can differ from encoding stop_text alone.
    let stop_token_sequences = stop_text.as_deref()
        .map(|text| pipeline.encode(text, false)).transpose()?.into_iter().collect();
    let control = GenerationControl { stop_token_sequences, ..Default::default() };
    let generation = SamplingGenerationConfig {
        max_new_tokens,
        eos_token_ids: pipeline.loaded.default_eos_token_ids.clone(),
        sampling: SamplingConfig { temperature: 0.8, top_k: 40, top_p: 0.9, seed: Some(42) },
    };
    let mut stdout = io::stdout().lock();
    let mut write_error = None;
    let result = generate_causal_sampled_stream(
        &pipeline.loaded.model, &(&pipeline.loaded.config).into(), &prompt_ids,
        &generation, &control, &device,
        |event| {
            let value = serde_json::json!({
                "event": "token", "token_id": event.token_id,
                "generated_tokens": event.generated_token_ids.len(),
                "natural_finish": event.finish_reason.map(|reason| format!("{reason:?}")),
            });
            if let Err(error) = writeln!(stdout, "{value}").and_then(|_| stdout.flush()) {
                write_error = Some(error);
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        },
    )?;
    if let Some(error) = write_error { return Err(error.into()); }
    writeln!(stdout, "{}", serde_json::json!({
        "event": "finished", "finish_reason": format!("{:?}", result.finish_reason),
        "generated_token_ids": result.output.generated_token_ids,
        "text": pipeline.decode(&result.output.generated_token_ids, true)?,
    }))?;
    Ok(())
}
