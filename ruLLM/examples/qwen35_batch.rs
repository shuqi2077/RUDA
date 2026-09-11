#[cfg(not(feature = "nvidia"))]
fn main() {
    eprintln!("qwen35_batch requires --features nvidia-ptx");
    std::process::exit(1);
}

#[cfg(feature = "nvidia")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use half::bf16;
    use ruda_tensor_device::cuda::{Cuda, CudaDevice};
    use rullm::*;
    use std::{collections::BTreeMap, io, path::Path, time::Instant};
    type B = Cuda<bf16, i32>;
    let mut args = std::env::args().skip(1);
    let directory = args.next().ok_or_else(|| {
        io::Error::other(
            "usage: qwen35_batch <model-directory> [greedy|sampled] [tokens>=4] [rounds>=1]",
        )
    })?;
    let mode = args.next().unwrap_or_else(|| "greedy".into());
    let n = args
        .next()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(8);
    let rounds = args
        .next()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    if !matches!(mode.as_str(), "greedy" | "sampled")
        || n < 4
        || rounds == 0
        || args.next().is_some()
    {
        return Err(io::Error::other("invalid arguments").into());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(Path::new(&directory).join("tokenizer.json"))
        .map_err(io::Error::other)?;
    let texts = [
        "The capital of France is",
        "The capital of Germany is",
        "Hello",
        "The capital of Italy is",
    ];
    let prompts = texts
        .iter()
        .map(|text| {
            tokenizer
                .encode(*text, false)
                .map_err(io::Error::other)
                .and_then(|v| {
                    v.get_ids()
                        .iter()
                        .map(|&id| i32::try_from(id).map_err(io::Error::other))
                        .collect::<Result<Vec<_>, _>>()
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        prompts[0].len(),
        prompts[1].len(),
        "initial requests must share a prefill bucket"
    );
    assert_ne!(
        prompts[0].len(),
        prompts[2].len(),
        "exercise different context lengths"
    );
    let device = CudaDevice::default();
    let start = Instant::now();
    let loaded = load_huggingface_qwen35_text::<B>(&directory, &device)?;
    println!(
        "{}",
        serde_json::json!({"stage":"loaded","seconds":start.elapsed().as_secs_f64()})
    );
    let limits = loaded.model.batch_limits();
    let config = GreedyGenerationConfig {
        max_new_tokens: n,
        eos_token_ids: vec![],
    };
    let sampling = SamplingGenerationConfig {
        max_new_tokens: n,
        eos_token_ids: vec![],
        sampling: SamplingConfig {
            seed: Some(42),
            ..Default::default()
        },
    };
    let mut references = Vec::new();
    let start = Instant::now();
    for (i, prompt) in prompts.iter().enumerate() {
        let request_start = Instant::now();
        let result = if mode == "sampled" {
            generate_causal_sampled(&loaded.model, &limits, prompt, &sampling, &device)?
        } else {
            generate_causal_greedy(&loaded.model, &limits, prompt, &config, &device)?
        };
        println!(
            "{}",
            serde_json::json!({"stage":"reference","request":i,"completed":i+1,"total":prompts.len(),
            "seconds":request_start.elapsed().as_secs_f64(),"tokens":result.generated_token_ids})
        );
        references.push(result.generated_token_ids);
    }
    let reference_seconds = start.elapsed().as_secs_f64();
    let max_context = prompts
        .iter()
        .map(Vec::len)
        .max()
        .unwrap()
        .checked_add(n)
        .ok_or_else(|| io::Error::other("length overflow"))?;
    let pages = 4 * max_context.div_ceil(2);
    let mut executor = DeviceBatchExecutor::new(
        &loaded.model,
        device.clone(),
        ContinuousBatchConfig {
            max_active_sequences: 4,
            max_batch_tokens: 64,
        },
        PagedKvCacheConfig {
            block_size: 2,
            num_pages: pages,
            max_sequence_length: max_context,
        },
        ContinuousBatchOptions::default(),
    )?;
    let total = rounds * (3 * n + 2) + 1;
    let mut completed = 0;
    let start = Instant::now();
    let mut max_batch = 0;
    let mut unequal_decode = false;
    for round in 0..rounds {
        let mut ids = BTreeMap::new();
        let mut submitted = BTreeMap::new();
        let mut first = BTreeMap::new();
        let mut output = BTreeMap::<RequestId, Vec<i32>>::new();
        for i in 0..2 {
            let id = if mode == "sampled" {
                executor.submit_sampled(prompts[i].clone(), sampling.clone())?
            } else {
                executor.submit(prompts[i].clone(), config.clone())?
            };
            ids.insert(i, id);
            submitted.insert(id, Instant::now());
        }
        let mut steps = 0;
        while !executor.is_idle() {
            let batch_start = Instant::now();
            let event = executor
                .step()?
                .ok_or_else(|| io::Error::other("non-idle executor made no progress"))?;
            steps += 1;
            assert!(steps <= 4 * n + 8, "unexpected scheduler stall");
            max_batch = max_batch.max(event.batch.batch_size());
            if event.batch.kind == ScheduledBatchKind::Decode
                && event
                    .batch
                    .sequences
                    .iter()
                    .any(|r| r.start_position != event.batch.sequences[0].start_position)
            {
                unequal_decode = true;
            }
            for (row, token) in event.batch.sequences.iter().zip(&event.generated_token_ids) {
                first
                    .entry(row.request_id)
                    .or_insert_with(|| submitted[&row.request_id].elapsed().as_secs_f64());
                output.entry(row.request_id).or_default().push(*token);
            }
            completed += event.generated_token_ids.len();
            println!(
                "{}",
                serde_json::json!({"stage":"batch","round":round,"kind":format!("{:?}",event.batch.kind),
                "batch_size":event.batch.batch_size(),"positions":event.batch.sequences.iter().map(|r| r.start_position).collect::<Vec<_>>(),
                "seconds":batch_start.elapsed().as_secs_f64(),"completed":completed,"total":total,
                "free_pages":executor.snapshot().free_kv_pages,"resident_layer_pages":executor.cache().resident_layer_pages(),
                "resident_requests":executor.cache().resident_requests()})
            );
            if steps == 1 {
                assert_eq!(event.batch.batch_size(), 2);
                let id = if mode == "sampled" {
                    executor.submit_sampled(prompts[2].clone(), sampling.clone())?
                } else {
                    executor.submit(prompts[2].clone(), config.clone())?
                };
                ids.insert(2, id);
                submitted.insert(id, Instant::now());
            }
            if steps == 2 {
                let b = ids[&1];
                let cancelled = executor
                    .cancel(b)?
                    .ok_or_else(|| io::Error::other("missing active cancellation"))?;
                assert_eq!(cancelled.generated_token_ids, references[1][..2]);
                assert_eq!(executor.cache().sequence_length(b), None);
                let id = if mode == "sampled" {
                    executor.submit_sampled(prompts[3].clone(), sampling.clone())?
                } else {
                    executor.submit(prompts[3].clone(), config.clone())?
                };
                ids.insert(3, id);
                submitted.insert(id, Instant::now());
            }
            while let Some(result) = executor.pop_finished() {
                let index = *ids
                    .iter()
                    .find(|(_, id)| **id == result.request_id)
                    .unwrap()
                    .0;
                assert_eq!(
                    result.generated_token_ids, references[index],
                    "request {index} differs from independent generation"
                );
                assert_eq!(output[&result.request_id], result.generated_token_ids);
                println!(
                    "{}",
                    serde_json::json!({"stage":"finished","round":round,"request":index,
                    "first_token_seconds":first[&result.request_id],"latency_seconds":submitted[&result.request_id].elapsed().as_secs_f64(),
                    "tokens":result.generated_token_ids})
                );
            }
        }
        assert_eq!(executor.snapshot().free_kv_pages, pages);
        assert_eq!(executor.cache().resident_layer_pages(), 0);
        assert_eq!(executor.cache().resident_requests(), 0);
    }
    // Reuse the emptied executor, then verify EOS and queued cancellation cleanup.
    let queued = executor.submit(prompts[1].clone(), config.clone())?;
    assert!(executor.cancel(queued)?.is_some());
    let id = executor.submit(
        prompts[0].clone(),
        GreedyGenerationConfig {
            max_new_tokens: n,
            eos_token_ids: vec![if mode == "greedy" {
                references[0][0]
            } else {
                generate_causal_greedy(
                    &loaded.model,
                    &limits,
                    &prompts[0],
                    &GreedyGenerationConfig {
                        max_new_tokens: 1,
                        eos_token_ids: vec![],
                    },
                    &device,
                )?
                .generated_token_ids[0]
            }],
        },
    )?;
    executor.step()?;
    let eos = executor
        .pop_finished()
        .ok_or_else(|| io::Error::other("EOS request did not finish"))?;
    assert_eq!(eos.request_id, id);
    assert!(eos.stopped_on_eos);
    assert_eq!(eos.generated_token_ids.len(), 1);
    completed += 1;
    assert!(executor.is_idle());
    assert_eq!(executor.snapshot().free_kv_pages, pages);
    assert_eq!(executor.cache().resident_layer_pages(), 0);
    assert_eq!(executor.cache().resident_requests(), 0);
    assert!(max_batch >= 2);
    assert!(unequal_decode);
    assert_eq!(completed, total);
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::json!({"stage":"verified","mode":mode,"rounds":rounds,"completed":completed,"total":total,
        "max_batch":max_batch,"unequal_context_decode":unequal_decode,"reference_seconds":reference_seconds,
        "batch_scenario_seconds":seconds,"scenario_tokens_per_second":completed as f64/seconds,
        "free_pages":executor.snapshot().free_kv_pages,"resident_layer_pages":executor.cache().resident_layer_pages()})
    );
    Ok(())
}
