use super::*;
use crate::{ContinuousBatchConfig, PagedKvCacheConfig, SamplingConfig};

fn scheduler() -> ContinuousBatchScheduler {
    ContinuousBatchScheduler::new(
        ContinuousBatchConfig {
            max_active_sequences: 4,
            max_batch_tokens: 8,
        },
        PagedKvCacheConfig {
            block_size: 2,
            num_pages: 32,
            max_sequence_length: 16,
        },
    )
    .unwrap()
}

fn config(seed: u64, max_new_tokens: usize) -> SamplingGenerationConfig {
    SamplingGenerationConfig {
        max_new_tokens,
        eos_token_ids: Vec::new(),
        sampling: SamplingConfig {
            seed: Some(seed),
            ..Default::default()
        },
    }
}

#[test]
fn mixed_batch_keeps_greedy_selection_and_request_local_sampling() {
    let mut scheduler = scheduler();
    scheduler
        .submit(
            vec![1],
            GreedyGenerationConfig {
                max_new_tokens: 1,
                eos_token_ids: vec![],
            },
        )
        .unwrap();
    scheduler.submit_sampled(vec![1], config(17, 1)).unwrap();
    let batch = scheduler.schedule().unwrap().unwrap();
    let row = [0.0, 1.0, 2.0];
    let expected = TokenSampler::new(config(17, 1).sampling)
        .unwrap()
        .sample(&row)
        .unwrap();
    let selection = scheduler
        .select_batch_tokens(batch.id, &[f32::NAN, 3.0, 3.0, 0.0, 1.0, 2.0], 3)
        .unwrap();
    assert_eq!(selection.tokens, vec![1, expected]);
    assert_eq!(
        scheduler.complete_selected_batch(selection).unwrap(),
        vec![1, expected]
    );
    assert_eq!(scheduler.snapshot().finished_requests, 2);
    assert_eq!(scheduler.snapshot().free_kv_pages, 32);
}

#[test]
fn failed_batch_and_stale_selection_do_not_consume_random_draws() {
    let mut scheduler = scheduler();
    scheduler.submit_sampled(vec![1], config(5, 3)).unwrap();
    let row = [0.0; 8];
    let mut reference = TokenSampler::new(config(5, 3).sampling).unwrap();
    let first = scheduler.schedule().unwrap().unwrap();
    let stale = scheduler
        .select_batch_tokens(first.id, &row, row.len())
        .unwrap();
    let expected_first = reference.sample(&row).unwrap();
    assert_eq!(stale.tokens, vec![expected_first]);
    scheduler.fail_batch(first.id).unwrap();
    let retry = scheduler.schedule().unwrap().unwrap();
    assert!(scheduler.complete_selected_batch(stale).is_err());
    let selection = scheduler
        .select_batch_tokens(retry.id, &row, row.len())
        .unwrap();
    assert_eq!(selection.tokens, vec![expected_first]);
    scheduler.complete_selected_batch(selection).unwrap();
    let next = scheduler.schedule().unwrap().unwrap();
    let selection = scheduler
        .select_batch_tokens(next.id, &row, row.len())
        .unwrap();
    assert_eq!(selection.tokens, vec![reference.sample(&row).unwrap()]);
}

#[test]
fn invalid_later_row_does_not_advance_an_earlier_sampler() {
    let mut scheduler = scheduler();
    scheduler.submit_sampled(vec![1], config(11, 2)).unwrap();
    scheduler.submit_sampled(vec![1], config(12, 2)).unwrap();
    let batch = scheduler.schedule().unwrap().unwrap();
    assert!(
        scheduler
            .select_batch_tokens(batch.id, &[0.0, 0.0, 0.0, f32::NAN], 2)
            .is_err()
    );
    let selection = scheduler
        .select_batch_tokens(batch.id, &[0.0; 4], 2)
        .unwrap();
    let expected: Vec<i32> = [11, 12]
        .into_iter()
        .map(|seed| {
            TokenSampler::new(config(seed, 2).sampling)
                .unwrap()
                .sample(&[0.0; 2])
                .unwrap()
        })
        .collect();
    assert_eq!(selection.tokens, expected);
}

#[test]
fn previews_are_repeatable_and_validate_shape_and_batch() {
    let mut scheduler = scheduler();
    scheduler.submit_sampled(vec![1], config(42, 2)).unwrap();
    let batch = scheduler.schedule().unwrap().unwrap();
    let first = scheduler
        .select_batch_tokens(batch.id, &[0.0; 4], 4)
        .unwrap();
    let second = scheduler
        .select_batch_tokens(batch.id, &[0.0; 4], 4)
        .unwrap();
    assert_eq!(first.tokens, second.tokens);
    assert!(
        scheduler
            .select_batch_tokens(batch.id + 1, &[0.0; 4], 4)
            .is_err()
    );
    assert!(
        scheduler
            .select_batch_tokens(batch.id, &[0.0; 3], 4)
            .is_err()
    );
    assert!(scheduler.select_batch_tokens(batch.id, &[], 0).is_err());
    scheduler.complete_selected_batch(second).unwrap();
    assert!(scheduler.complete_selected_batch(first).is_err());
}

#[test]
fn invalid_sampling_options_do_not_enqueue_or_allocate_a_request_id() {
    let mut scheduler = scheduler();
    let before = scheduler.snapshot();
    let next_id = scheduler.next_request;
    let mut invalid = config(1, 2);
    invalid.sampling.temperature = 0.0;
    assert!(scheduler.submit_sampled(vec![1], invalid).is_err());
    assert_eq!(scheduler.snapshot(), before);
    assert_eq!(scheduler.next_request, next_id);
}

#[test]
fn zero_limit_and_eos_release_sampled_requests() {
    let mut scheduler = scheduler();
    let empty = scheduler.submit_sampled(vec![1], config(1, 0)).unwrap();
    let finished = scheduler.pop_finished().unwrap();
    assert_eq!(finished.request_id, empty);
    assert!(finished.generated_token_ids.is_empty());
    let mut generation = config(1, 4);
    generation.eos_token_ids = vec![1];
    scheduler.submit_sampled(vec![1], generation).unwrap();
    let batch = scheduler.schedule().unwrap().unwrap();
    let selection = scheduler
        .select_batch_tokens(batch.id, &[f32::NEG_INFINITY, 0.0], 2)
        .unwrap();
    scheduler.complete_selected_batch(selection).unwrap();
    let finished = scheduler.pop_finished().unwrap();
    assert_eq!(finished.generated_token_ids, vec![1]);
    assert!(finished.stopped_on_eos);
    assert_eq!(scheduler.snapshot().free_kv_pages, 32);
}

#[test]
fn interleaved_arrivals_and_finishes_do_not_change_request_rng_sequence() {
    fn run(interleave: bool) -> Vec<i32> {
        let mut scheduler = scheduler();
        let primary = scheduler.submit_sampled(vec![1], config(123, 5)).unwrap();
        let mut added = false;
        let mut result = None;
        while let Some(batch) = scheduler.schedule().unwrap() {
            let logits = vec![0.0; batch.batch_size() * 8];
            let selection = scheduler.select_batch_tokens(batch.id, &logits, 8).unwrap();
            scheduler.complete_selected_batch(selection).unwrap();
            if interleave && !added {
                scheduler
                    .submit_sampled(vec![1, 2], config(456, 2))
                    .unwrap();
                added = true;
            }
            while let Some(finished) = scheduler.pop_finished() {
                if finished.request_id == primary {
                    result = Some(finished.generated_token_ids);
                }
            }
        }
        assert_eq!(scheduler.snapshot().free_kv_pages, 32);
        result.unwrap()
    }
    assert_eq!(run(false), run(true));
}
