#[cfg(not(feature = "nvidia"))]
fn main() { eprintln!("qwen35_speculative requires --features nvidia-ptx"); std::process::exit(1); }

#[cfg(feature = "nvidia")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use half::bf16;
    use ruda_tensor::api::{DType, Int, Tensor, TensorData};
    use ruda_tensor_device::cuda::{Cuda, CudaDevice};
    use rullm::*;
    use std::{io, path::Path, time::Instant, ops::ControlFlow};
    type B = Cuda<bf16, i32>;
    let mut args = std::env::args().skip(1);
    let directory = args.next().ok_or_else(|| io::Error::other(
        "usage: qwen35_speculative <model-directory> <prompt> [max-new-tokens] [draft-tokens] [greedy|sampled]"))?;
    let prompt = args.next().ok_or_else(|| io::Error::other("prompt required"))?;
    let max_new_tokens = args.next().map(|s| s.parse()).transpose()?.unwrap_or(8);
    let draft_tokens = args.next().map(|s| s.parse()).transpose()?.unwrap_or(3);
    let mode = args.next().unwrap_or_else(|| "greedy".into());
    if args.next().is_some() || !matches!(mode.as_str(), "greedy" | "sampled") {
        return Err(io::Error::other("invalid arguments").into());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(Path::new(&directory).join("tokenizer.json"))
        .map_err(io::Error::other)?;
    let ids = tokenizer.encode(prompt, false).map_err(io::Error::other)?.get_ids().iter()
        .map(|&id| i32::try_from(id)).collect::<Result<Vec<_>, _>>()?;
    let device = CudaDevice::default();
    let start = Instant::now();
    let loaded = load_huggingface_qwen35_text::<B>(&directory, &device)?;
    println!("{}", serde_json::json!({"stage":"loaded","seconds":start.elapsed().as_secs_f64(),"prompt_token_ids":ids}));
    let limits = CausalModelLimits { vocab_size: loaded.model.config.vocab_size, max_sequence_length: loaded.model.config.max_position_embeddings };
    let config = SpeculativeGenerationConfig {
        max_new_tokens, draft_tokens, eos_token_ids: vec![loaded.model.config.eos_token_id],
        sampling: (mode == "sampled").then_some(SamplingConfig { seed: Some(42), ..SamplingConfig::default() }),
    };
    let start = Instant::now();
    let result = generate_causal_speculative_stream(
        &loaded.model, &limits, &loaded.model, &limits, &ids, &config,
        &GenerationControl::default(), &device, |event| {
            println!("{}", serde_json::json!({"stage":"token","completed":event.generated_token_ids.len(),
                "total":max_new_tokens,"token_id":event.token_id}));
            ControlFlow::Continue(())
        },
    )?;
    let seconds = start.elapsed().as_secs_f64();
    assert!(!result.output.generated_token_ids.is_empty() || max_new_tokens == 0);
    assert!(result.output.generated_token_ids.len() <= max_new_tokens);
    if max_new_tokens > 0 { assert!(result.stats.target_block_forwards > 0); }
    let tokens = result.output.generated_token_ids.iter().map(|&id| id as u32).collect::<Vec<_>>();
    println!("{}", serde_json::json!({"stage":"generated","mode":mode,"seconds":seconds,
        "generated_token_ids":tokens,"text":tokenizer.decode(&tokens,true).map_err(io::Error::other)?,
        "rounds":result.stats.rounds,"proposed":result.stats.proposed_tokens,"accepted":result.stats.accepted_tokens,
        "rejected_blocks":result.stats.rejected_blocks,"bonus_tokens":result.stats.bonus_tokens,
        "target_block_forwards":result.stats.target_block_forwards,"replayed_prefix_tokens":result.stats.replayed_prefix_tokens,
        "finish_reason":format!("{:?}",result.finish_reason)}));

    // Verify that an abandoned block does not mutate a fork's attention KV,
    // convolution history or recurrent state. Both continuations use the same
    // retained prefix, with no quality/reference-model requirement.
    if let Some(&token) = result.output.generated_token_ids.first() {
        let input = || Tensor::<B,2,Int>::from_data(TensorData::new(ids.clone(), [1,ids.len()]), &device);
        let step = || Tensor::<B,2,Int>::from_data([[token]], &device);
        let mut original = loaded.model.new_cache();
        let _ = loaded.model.forward_cached_last(input(), &mut original)?;
        let saved = loaded.model.fork_cache(&original);
        let mut reference = loaded.model.new_cache();
        let _ = loaded.model.forward_cached_last(input(), &mut reference)?;
        let a = loaded.model.forward_cached_last(step(), &mut reference)?.cast(DType::F32).into_data().to_vec::<f32>()?;
        let mut abandoned = loaded.model.fork_cache(&original);
        let _ = loaded.model.forward_cached(Tensor::<B,2,Int>::from_data([[token,token]], &device), &mut abandoned)?.into_data();
        let mut restored = saved;
        let b = loaded.model.forward_cached_last(step(), &mut restored)?.cast(DType::F32).into_data().to_vec::<f32>()?;
        assert_eq!(original.sequence_length(), ids.len());
        assert_eq!(reference.sequence_length(), ids.len()+1);
        assert_eq!(restored.sequence_length(), ids.len()+1);
        assert_eq!(a, b, "abandoned block changed the saved Qwen cache state");
        println!("{}", serde_json::json!({"stage":"cache_restore_verified","sequence_length":restored.sequence_length()}));
    }
    Ok(())
}
