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
