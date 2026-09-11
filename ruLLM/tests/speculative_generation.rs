use ruda_tensor::api::{Int, Tensor, TensorData};
use ruda_tensor_host::{Host, HostDevice};
use rullm::*;
use std::{cell::RefCell, ops::ControlFlow};

// A state-dependent CPU test double checks control flow and cache branches;
// the separate Qwen example exercises the real model and GPU kernels.
#[derive(Default)]
struct Model {
    wrong_at: Option<usize>,
    probabilities: Option<Vec<f32>>,
    calls: RefCell<Vec<(bool, Vec<i32>, Vec<i32>)>>,
}

impl Model {
    fn row(&self, cache: &[i32]) -> Vec<f32> {
        if let Some(row) = &self.probabilities { return row.clone(); }
        let next = ((cache.iter().sum::<i32>() + 1) % 5) as usize;
        let next = if self.wrong_at.is_some_and(|n| cache.len() % 3 == n) { (next + 1) % 5 } else { next };
        let mut row = vec![-10.0; 5]; row[next] = 10.0; row
    }
    fn forward(&self, tokens: Tensor<Host, 2, Int>, cache: &mut Vec<i32>, block: bool) -> Tensor<Host, 3> {
        let input = tokens.into_data().convert::<i32>().to_vec::<i32>().unwrap();
        self.calls.borrow_mut().push((block, cache.clone(), input.clone()));
        let mut rows = Vec::new();
        for token in &input { cache.push(*token); rows.extend(self.row(cache)); }
        let v = self.row(cache).len();
        let count = if block { input.len() } else { rows = rows[rows.len() - v..].to_vec(); 1 };
        Tensor::from_data(TensorData::new(rows, [1, count, v]), &HostDevice)
    }
}
impl CausalModel<Host> for Model {
    type Cache = Vec<i32>;
    fn new_cache(&self) -> Self::Cache { Vec::new() }
    fn forward_cached_last(&self, tokens: Tensor<Host, 2, Int>, cache: &mut Self::Cache) -> Tensor<Host, 3> {
        self.forward(tokens, cache, false)
    }
}
impl SpeculativeModel<Host> for Model {
    fn fork_cache(&self, cache: &Self::Cache) -> Self::Cache { cache.clone() }
    fn try_forward_cached_block(&self, tokens: Tensor<Host, 2, Int>, cache: &mut Self::Cache)
        -> Result<Tensor<Host, 3>, GenerationError> { Ok(self.forward(tokens, cache, true)) }
}
fn limits() -> CausalModelLimits { CausalModelLimits { vocab_size: 5, max_sequence_length: 64 } }
fn config(n: usize) -> SpeculativeGenerationConfig {
    SpeculativeGenerationConfig { max_new_tokens: n, eos_token_ids: vec![], draft_tokens: 3, sampling: None }
}
fn run(target: &Model, draft: &Model, c: &SpeculativeGenerationConfig) -> SpeculativeGenerationOutput {
    generate_causal_speculative(target, &limits(), draft, &limits(), &[1], c, &HostDevice).unwrap()
}

#[test]
fn self_draft_uses_batched_verification_and_bonus_without_exceeding_length() {
    for n in [0, 1, 2, 3, 4, 8, 11] {
        let model = Model::default();
        let result = run(&model, &model, &config(n));
        let baseline = generate_causal_greedy(&model, &limits(), &[1],
            &GreedyGenerationConfig { max_new_tokens: n, eos_token_ids: vec![] }, &HostDevice).unwrap();
        assert_eq!(result.output, baseline);
        assert_eq!(result.stats.rejected_blocks, 0);
        assert_eq!(result.output.generated_token_ids.len(), n);
        if n >= 4 { assert!(result.stats.bonus_tokens > 0); }
        if n >= 3 { assert!(model.calls.borrow().iter().any(|(block, _, tokens)| *block && tokens.len() == 3)); }
    }
}

#[test]
fn rejection_restores_both_prefixes_and_never_replays_rejected_tokens() {
    for wrong_at in 0..3 {
        let target = Model::default();
        let draft = Model { wrong_at: Some(wrong_at), ..Model::default() };
        let result = run(&target, &draft, &config(15));
        let baseline = generate_causal_greedy(&Model::default(), &limits(), &[1],
            &GreedyGenerationConfig { max_new_tokens: 15, eos_token_ids: vec![] }, &HostDevice).unwrap();
        assert_eq!(result.output, baseline);
        assert!(result.stats.rejected_blocks > 0);
        assert!(result.stats.replayed_prefix_tokens > 0);
        for (_, prefix, _) in target.calls.borrow().iter() {
            assert_eq!(prefix, &baseline.token_ids[..prefix.len()]);
        }
        for (_, prefix, _) in draft.calls.borrow().iter().filter(|(_, prefix, _)| prefix.len() == 1) {
            assert_eq!(prefix, &[1]);
        }
    }
}

#[test]
fn accepted_eos_and_stop_sequences_emit_no_bonus_or_unverified_suffix() {
    let model = Model::default();
    let mut c = config(12); c.eos_token_ids = vec![2];
    let eos = run(&model, &model, &c);
    assert_eq!(eos.output.generated_token_ids, [2]);
    assert_eq!(eos.finish_reason, GenerationFinishReason::EosToken(2));
    assert_eq!(eos.stats.bonus_tokens, 0);
    c.eos_token_ids.clear();
    let mut callbacks = Vec::new();
    let stop = generate_causal_speculative_stream(&model, &limits(), &model, &limits(), &[1], &c,
        &GenerationControl { stop_token_sequences: vec![vec![2, 4]], cancellation: None }, &HostDevice,
        |event| { callbacks.push(event.token_id); ControlFlow::Continue(()) }).unwrap();
    assert_eq!(stop.output.generated_token_ids, [2, 4]);
    assert_eq!(callbacks, [2, 4]);
    assert_eq!(stop.finish_reason, GenerationFinishReason::StopSequence(0));
}

#[test]
fn callback_and_pre_cancel_preserve_existing_stop_order() {
    let model = Model::default();
    let result = generate_causal_speculative_stream(&model, &limits(), &model, &limits(), &[1], &config(10),
        &GenerationControl::default(), &HostDevice, |_| ControlFlow::Break(())).unwrap();
    assert_eq!(result.output.generated_token_ids, [2]);
    assert_eq!(result.finish_reason, GenerationFinishReason::Cancelled);
    let flag = GenerationCancellation::new(); flag.cancel();
    let model = Model::default();
    let result = generate_causal_speculative_stream(&model, &limits(), &model, &limits(), &[1], &config(10),
        &GenerationControl { cancellation: Some(flag), stop_token_sequences: vec![] }, &HostDevice,
        |_| panic!("cancelled request must not emit")).unwrap();
    assert!(result.output.generated_token_ids.is_empty());
    assert!(model.calls.borrow().is_empty());
}

#[test]
fn invalid_inputs_are_rejected_before_model_execution() {
    let model = Model::default();
    let mut c = config(4); c.draft_tokens = 0;
    assert!(generate_causal_speculative(&model, &limits(), &model, &limits(), &[1], &c, &HostDevice).is_err());
    c.draft_tokens = 2;
    let mut mismatch = limits(); mismatch.vocab_size = 6;
    assert!(generate_causal_speculative(&model, &limits(), &model, &mismatch, &[1], &c, &HostDevice).is_err());
    assert!(generate_causal_speculative(&model, &limits(), &model, &limits(), &[], &c, &HostDevice).is_err());
    c.max_new_tokens = usize::MAX;
    assert!(generate_causal_speculative(&model, &limits(), &model, &limits(), &[1], &c, &HostDevice).is_err());
    assert!(model.calls.borrow().is_empty());
}

#[test]
fn sampled_rejection_matches_target_distribution_and_seed_is_repeatable() {
    let target = Model { probabilities: Some(vec![0.1f32.ln(), 0.2f32.ln(), 0.7f32.ln(), f32::NEG_INFINITY, f32::NEG_INFINITY]), ..Model::default() };
    let draft = Model { probabilities: Some(vec![0.7f32.ln(), 0.2f32.ln(), 0.1f32.ln(), f32::NEG_INFINITY, f32::NEG_INFINITY]), ..Model::default() };
    let mut c = config(1);
    let mut counts = [0; 5];
    for seed in 0..2000 {
        c.sampling = Some(SamplingConfig { seed: Some(seed), ..SamplingConfig::default() });
        let a = run(&target, &draft, &c);
        if seed < 5 { assert_eq!(a, run(&target, &draft, &c)); }
        counts[a.output.generated_token_ids[0] as usize] += 1;
    }
    for (count, expected) in counts.iter().zip([0.1, 0.2, 0.7, 0.0, 0.0]) {
        assert!((*count as f64 / 2000.0 - expected).abs() < 0.04, "{counts:?}");
    }
}
