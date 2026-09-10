//! Host-side sampling microbenchmark against the pre-optimization implementation.
//! This does not measure model execution, device transfer or end-to-end latency.
use chacha20::ChaCha12Rng;
use ruda_core::rand::{RngExt, SeedableRng};
use rullm::{GenerationError, SamplingConfig, TokenSampler};
use std::{error::Error, hint::black_box, io, time::Instant};

fn legacy_probabilities(
    logits: &[f32],
    config: SamplingConfig,
) -> Result<Vec<f64>, GenerationError> {
    config.validate()?;
    if logits.is_empty() || logits.len() - 1 > i32::MAX as usize {
        return Err(GenerationError(
            "sampling requires a nonempty i32-indexed vocabulary".into(),
        ));
    }
    if logits
        .iter()
        .any(|value| value.is_nan() || *value == f32::INFINITY)
    {
        return Err(GenerationError(
            "sampling logits contain NaN or positive infinity".into(),
        ));
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if maximum == f32::NEG_INFINITY {
        return Err(GenerationError("all sampling logits are masked".into()));
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    if config.top_k > 0 || config.top_p < 1.0 {
        // Equal logits have a deterministic token-ID order for nucleus filtering.
        order.sort_by(|&left, &right| {
            logits[left]
                .partial_cmp(&logits[right])
                .unwrap()
                .then(left.cmp(&right))
        });
    }
    let threshold = if config.top_k > 0 {
        logits[order[logits.len() - config.top_k.min(logits.len())]]
    } else {
        f32::NEG_INFINITY
    };
    let mut probabilities: Vec<f64> = logits
        .iter()
        .map(|&value| {
            if value < threshold {
                0.0
            } else {
                ((value as f64 - maximum as f64) / config.temperature).exp()
            }
        })
        .collect();
    let total: f64 = probabilities.iter().sum();
    if config.top_p < 1.0 {
        let mut cumulative = 0.0;
        for &token in order.iter().take(order.len() - 1) {
            cumulative += probabilities[token] / total;
            if cumulative <= 1.0 - config.top_p {
                probabilities[token] = 0.0;
            }
        }
    }
    let retained: f64 = probabilities.iter().sum();
    for probability in &mut probabilities {
        *probability /= retained;
    }
    Ok(probabilities)
}

fn legacy_sample(logits: &[f32], config: SamplingConfig, rng: &mut ChaCha12Rng) -> i32 {
    let distribution = legacy_probabilities(logits, config).unwrap();
    let draw = rng.random::<f64>();
    let mut sum = 0.0;
    let mut last = 0;
    for (id, probability) in distribution.into_iter().enumerate() {
        if probability == 0.0 { continue; }
        last = id as i32;
        sum += probability;
        if draw < sum { break; }
    }
    last
}

fn measure(mut sample: impl FnMut() -> i32, steps: usize) -> (f64, u64) {
    for _ in 0..8 { black_box(sample()); }
    let started = Instant::now();
    let mut checksum = 0_u64;
    for _ in 0..steps { checksum = checksum.wrapping_add(black_box(sample()) as u64); }
    (started.elapsed().as_nanos() as f64 / steps as f64, checksum)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let vocabulary = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(32_000);
    let steps = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(100);
    if vocabulary == 0 || vocabulary > 1_000_000 || steps == 0 || args.next().is_some() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput,
            "usage: sampling_bench [vocabulary: 1..1000000] [positive-steps]").into());
    }
    let mut state = 42_u64;
    let logits: Vec<f32> = (0..vocabulary).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 32) as i32 % 10000) as f32 / 1000.0
    }).collect();
    for (name, top_k, top_p) in [
        ("unfiltered", 0, 1.0), ("top_k", 40, 1.0),
        ("top_p", 0, 0.9), ("top_k_top_p", 40, 0.9),
    ] {
        let config = SamplingConfig { top_k, top_p, seed: Some(42), ..Default::default() };
        let mut baseline_times = Vec::new();
        let mut optimized_times = Vec::new();
        for repeat in 0..7 {
            let baseline = || {
                let mut rng = ChaCha12Rng::seed_from_u64(42);
                measure(|| legacy_sample(black_box(&logits), config, &mut rng), steps)
            };
            let optimized = || {
                let mut sampler = TokenSampler::new(config).unwrap();
                measure(|| sampler.sample(black_box(&logits)).unwrap(), steps)
            };
            let (before, after) = if repeat % 2 == 0 {
                (baseline(), optimized())
            } else {
                let after = optimized();
                (baseline(), after)
            };
            assert_eq!(before.1, after.1, "sampling checksums differ");
            baseline_times.push(before.0);
            optimized_times.push(after.0);
        }
        baseline_times.sort_by(f64::total_cmp);
        optimized_times.sort_by(f64::total_cmp);
        println!("{}", serde_json::json!({
            "case": name, "vocabulary": vocabulary, "steps_per_repeat": steps, "repeats": 7,
            "baseline_ns_per_token_median": baseline_times[3],
            "optimized_ns_per_token_median": optimized_times[3],
            "speedup": baseline_times[3] / optimized_times[3],
            "scope": "host sampling only; excludes model and device transfers",
        }));
    }
    Ok(())
}
