use super::*;

#[test]
fn seeded_stream_matches_existing_std_rng() {
    use ruda_core::rand::StdRng;

    for seed in [0, 1, 42, u64::MAX] {
        let mut original = StdRng::seed_from_u64(seed);
        let mut sampler = TokenSampler::new(SamplingConfig {
            seed: Some(seed),
            ..Default::default()
        }).unwrap();
        for _ in 0..4096 {
            assert_eq!(original.random::<f64>(), sampler.rng.random::<f64>());
        }
        let mut cloned = sampler.clone();
        for _ in 0..257 {
            assert_eq!(sampler.rng.random::<f64>(), cloned.rng.random::<f64>());
        }
    }
}

fn probabilities(logits: &[f32], temperature: f64, top_k: usize, top_p: f64) -> Vec<f64> {
    filtered_probabilities(
        logits,
        SamplingConfig {
            temperature,
            top_k,
            top_p,
            seed: Some(7),
        },
    )
    .unwrap()
}

#[test]
fn temperature_rescales_log_odds() {
    let distribution = probabilities(&[0.0, 2.0], 2.0, 0, 1.0);
    assert!((distribution[1] / distribution[0] - std::f64::consts::E).abs() < 1e-12);
}

#[test]
fn top_k_retains_threshold_ties_and_clamps_to_vocabulary() {
    assert_eq!(
        probabilities(&[-1.0, 2.0, 2.0], 1.0, 1, 1.0),
        vec![0.0, 0.5, 0.5]
    );
    assert_eq!(
        probabilities(&[0.0, 1.0], 1.0, usize::MAX, 1.0),
        probabilities(&[0.0, 1.0], 1.0, 0, 1.0)
    );
}

#[test]
fn nucleus_retains_boundary_token_and_at_least_one() {
    assert_eq!(
        probabilities(&[0.0; 4], 1.0, 0, 0.5),
        vec![0.0, 0.0, 0.5, 0.5]
    );
    assert_eq!(
        probabilities(&[0.0; 4], 1.0, 0, 0.01),
        vec![0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn top_k_is_applied_before_nucleus() {
    let distribution = probabilities(&[0.0, 1.0, 2.0], 1.0, 2, 0.7);
    assert_eq!(distribution, vec![0.0, 0.0, 1.0]);
}

#[test]
fn signed_zero_ties_use_token_order() {
    assert_eq!(probabilities(&[0.0, -0.0], 1.0, 0, 0.5), vec![0.0, 1.0]);
}

#[test]
fn masked_tokens_and_extreme_temperatures_are_handled() {
    assert_eq!(
        probabilities(&[f32::NEG_INFINITY, f32::MAX], f64::MIN_POSITIVE, 0, 1.0),
        vec![0.0, 1.0]
    );
    let distribution = probabilities(&[-f32::MAX, f32::MAX], f64::MAX, 0, 1.0);
    assert!((distribution[0] - 0.5).abs() < 1e-12);
}

#[test]
fn invalid_configuration_and_invalid_logits_are_rejected() {
    for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            TokenSampler::new(SamplingConfig {
                temperature,
                ..Default::default()
            })
            .is_err()
        );
    }
    for top_p in [0.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(
            TokenSampler::new(SamplingConfig {
                top_p,
                ..Default::default()
            })
            .is_err()
        );
    }
    let mut sampler = TokenSampler::new(SamplingConfig::default()).unwrap();
    for logits in [
        vec![],
        vec![f32::NAN],
        vec![f32::NEG_INFINITY; 3],
        vec![0.0, f32::INFINITY],
    ] {
        assert!(sampler.sample(&logits).is_err());
    }
}

#[test]
fn seed_and_cloned_state_are_reproducible() {
    let config = SamplingConfig {
        seed: Some(42),
        ..Default::default()
    };
    let mut first = TokenSampler::new(config).unwrap();
    let mut second = TokenSampler::new(config).unwrap();
    for _ in 0..128 {
        assert_eq!(first.sample(&[0.0; 5]), second.sample(&[0.0; 5]));
    }
    let mut cloned = first.clone();
    for _ in 0..128 {
        assert_eq!(first.sample(&[0.0; 5]), cloned.sample(&[0.0; 5]));
    }
}

#[test]
fn sampled_tokens_cannot_escape_filtered_support() {
    let mut sampler = TokenSampler::new(SamplingConfig {
        top_k: 2,
        seed: Some(5),
        ..Default::default()
    })
    .unwrap();
    let mut counts = [0_usize; 3];
    for _ in 0..1024 {
        counts[sampler.sample(&[-100.0, 1.0, 1.0]).unwrap() as usize] += 1;
    }
    assert_eq!(counts[0], 0);
    assert!(counts[1] > 0 && counts[2] > 0);
}

// Frozen pre-optimization reference, intentionally independent of workspace code.
fn reference_filtered_probabilities(
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

#[test]
fn optimized_probabilities_match_full_sort_reference_bit_for_bit() {
    let mut state = 42_u64;
    for size in [1, 2, 3, 4, 17, 65, 257] {
        for round in 0..24 {
            let mut logits: Vec<f32> = (0..size).map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                match state % 11 {
                    0 => f32::NEG_INFINITY,
                    1 => -0.0,
                    _ => ((state >> 32) as i32 % 32) as f32 / 3.0,
                }
            }).collect();
            logits[round % size] = 0.0;
            for temperature in [f64::MIN_POSITIVE, 0.7, 1.0, 2.0, f64::MAX] {
                for top_k in [0, 1, 2, size, usize::MAX] {
                    for top_p in [0.01, 0.5, 0.9, 1.0] {
                        let config = SamplingConfig { temperature, top_k, top_p, seed: Some(9) };
                        let actual = filtered_probabilities(&logits, config).unwrap();
                        let expected = reference_filtered_probabilities(&logits, config).unwrap();
                        let bits = |values: Vec<f64>| values.into_iter().map(f64::to_bits).collect::<Vec<_>>();
                        assert_eq!(bits(actual), bits(expected), "size={size}, {config:?}");
                    }
                }
            }
        }
    }
}

#[test]
fn workspace_reuses_storage_and_handles_growing_and_shrinking_rows() {
    let config = SamplingConfig { top_k: 2, top_p: 0.9, ..Default::default() };
    let mut workspace = SamplingWorkspace::default();
    let row = vec![1.0; 64];
    let initial = workspace.fill(&row, config).unwrap().as_ptr();
    assert_eq!(initial, workspace.fill(&row, config).unwrap().as_ptr());
    for size in [1, 17, 64, 257, 3, 64] {
        let row: Vec<f32> = (0..size).map(|i| (i % 7) as f32).collect();
        assert_eq!(workspace.fill(&row, config).unwrap(), reference_filtered_probabilities(&row, config).unwrap());
    }
}

#[test]
fn invalid_logits_do_not_advance_random_state_after_buffer_reuse() {
    let mut sampler = TokenSampler::new(SamplingConfig { seed: Some(42), ..Default::default() }).unwrap();
    sampler.sample(&[0.0; 16]).unwrap();
    let mut reference = sampler.clone();
    for row in [vec![], vec![f32::NAN], vec![f32::INFINITY], vec![f32::NEG_INFINITY; 5]] {
        assert!(sampler.sample(&row).is_err());
    }
    for _ in 0..256 {
        assert_eq!(sampler.sample(&[0.0; 8]).unwrap(), reference.sample(&[0.0; 8]).unwrap());
    }
}

#[test]
fn seeded_samples_match_the_legacy_cdf() {
    use ruda_core::rand::StdRng;
    for top_k in [0, 1, 5, 100] {
        for top_p in [0.2, 0.9, 1.0] {
            let config = SamplingConfig { top_k, top_p, seed: Some(17), ..Default::default() };
            let mut sampler = TokenSampler::new(config).unwrap();
            let mut rng = StdRng::seed_from_u64(17);
            for step in 0..128 {
                let row: Vec<f32> = (0..67).map(|i| ((i * 7 + step) % 19) as f32).collect();
                let probabilities = reference_filtered_probabilities(&row, config).unwrap();
                let draw = rng.random::<f64>();
                let mut cumulative = 0.0;
                let mut expected = None;
                for (id, probability) in probabilities.into_iter().enumerate() {
                    if probability == 0.0 { continue; }
                    expected = Some(id as i32);
                    cumulative += probability;
                    if draw < cumulative { break; }
                }
                assert_eq!(sampler.sample(&row).unwrap(), expected.unwrap());
            }
        }
    }
}

#[test]
fn clearing_workspace_preserves_random_stream() {
    let mut sampler = TokenSampler::new(SamplingConfig {
        seed: Some(42), top_k: 3, top_p: 0.9, ..Default::default()
    }).unwrap();
    sampler.sample(&[0.0; 16]).unwrap();
    let mut reference = sampler.clone();
    sampler.clear_workspace();
    for _ in 0..128 {
        assert_eq!(sampler.sample(&[0.0; 16]).unwrap(), reference.sample(&[0.0; 16]).unwrap());
    }
}
