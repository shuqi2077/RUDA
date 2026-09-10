//! CPU-only tests of the real shared generation loop. No model files or GPU.
use ruda_tensor::api::{Int, Tensor, TensorData};
use ruda_tensor_host::{Host, HostDevice};
use rullm::*;
use std::{cell::{Cell, RefCell}, ops::ControlFlow};

struct ScriptedModel {
    script: Vec<i32>,
    inputs: RefCell<Vec<Vec<i32>>>,
    caches: Cell<usize>,
    vocabulary: usize,
    forced_selection: Option<i32>,
    fail_forward: bool,
}

impl ScriptedModel {
    fn new(script: &[i32]) -> Self {
        Self {
            script: script.to_vec(), inputs: RefCell::new(Vec::new()), caches: Cell::new(0),
            vocabulary: 8, forced_selection: None, fail_forward: false,
        }
    }
}

impl CausalModel<Host> for ScriptedModel {
    type Cache = usize;

    fn new_cache(&self) -> usize {
        self.caches.set(self.caches.get() + 1);
        0
    }

    fn forward_cached_last(&self, tokens: Tensor<Host, 2, Int>, cache: &mut usize) -> Tensor<Host, 3> {
        self.inputs.borrow_mut().push(
            tokens.into_data().convert::<i32>().to_vec::<i32>().unwrap()
        );
        let token = self.script[*cache % self.script.len()];
        *cache += 1;
        let mut row = vec![f32::NEG_INFINITY; self.vocabulary];
        row[token as usize] = 1.0;
        Tensor::from_data(TensorData::new(row, [1, 1, self.vocabulary]), &HostDevice)
    }

    fn try_forward_cached_last(
        &self, tokens: Tensor<Host, 2, Int>, cache: &mut usize,
    ) -> Result<Tensor<Host, 3>, GenerationError> {
        if self.fail_forward { return Err(GenerationError("scripted forward failure".into())); }
        Ok(self.forward_cached_last(tokens, cache))
    }

    fn greedy_token(&self, logits: Tensor<Host, 3>) -> Result<i32, GenerationError> {
        if let Some(token) = self.forced_selection { return Ok(token); }
        let row = logits.into_data().to_vec::<f32>().unwrap();
        Ok(row.iter().position(|value| *value == 1.0).unwrap() as i32)
    }
}

fn limits() -> CausalModelLimits {
    CausalModelLimits { vocab_size: 8, max_sequence_length: 32 }
}

fn config(max_new_tokens: usize) -> GreedyGenerationConfig {
    GreedyGenerationConfig { max_new_tokens, eos_token_ids: vec![] }
}

#[test]
fn callbacks_observe_each_token_and_only_one_prompt_prefill_occurs() {
    let model = ScriptedModel::new(&[2, 3, 4]);
    let mut events = Vec::new();
    let result = generate_causal_greedy_stream(
        &model, &limits(), &[1, 7], &config(3), &GenerationControl::default(), &HostDevice,
        |event| {
            events.push((event.token_id, event.generated_token_ids.to_vec(), event.finish_reason));
            ControlFlow::Continue(())
        },
    ).unwrap();
    assert_eq!(result.output.token_ids, vec![1, 7, 2, 3, 4]);
    assert_eq!(result.finish_reason, GenerationFinishReason::MaxNewTokens);
    assert_eq!(events.len(), 3);
    assert_eq!(events[1].1, vec![2, 3]);
    assert_eq!(events[2].2, Some(GenerationFinishReason::MaxNewTokens));
    assert_eq!(*model.inputs.borrow(), vec![vec![1, 7], vec![2], vec![3]]);
    assert_eq!(model.caches.get(), 1);
}

#[test]
fn callback_break_keeps_last_token_and_avoids_an_extra_forward() {
    let model = ScriptedModel::new(&[2, 3, 4]);
    let result = generate_causal_greedy_stream(
        &model, &limits(), &[1], &config(8), &GenerationControl::default(), &HostDevice,
        |_| ControlFlow::Break(()),
    ).unwrap();
    assert_eq!(result.output.generated_token_ids, vec![2]);
    assert_eq!(result.finish_reason, GenerationFinishReason::Cancelled);
    assert_eq!(model.inputs.borrow().len(), 1);
}

#[test]
fn pre_cancelled_request_does_not_create_cache_or_execute_model() {
    let model = ScriptedModel::new(&[2]);
    let cancellation = GenerationCancellation::new();
    cancellation.cancel();
    let control = GenerationControl { cancellation: Some(cancellation), ..Default::default() };
    let result = generate_causal_greedy_stream(
        &model, &limits(), &[1], &config(8), &control, &HostDevice,
        |_| panic!("pre-cancelled request must not stream"),
    ).unwrap();
    assert_eq!(result.finish_reason, GenerationFinishReason::Cancelled);
    assert_eq!(result.output.token_ids, vec![1]);
    assert_eq!(model.caches.get(), 0);
}

#[test]
fn cancellation_flag_is_observed_before_next_forward() {
    let model = ScriptedModel::new(&[2]);
    let cancellation = GenerationCancellation::new();
    let control = GenerationControl { cancellation: Some(cancellation.clone()), ..Default::default() };
    let result = generate_causal_greedy_stream(
        &model, &limits(), &[1], &config(8), &control, &HostDevice,
        |_| { cancellation.cancel(); ControlFlow::Continue(()) },
    ).unwrap();
    assert_eq!(result.finish_reason, GenerationFinishReason::Cancelled);
    assert_eq!(model.inputs.borrow().len(), 1);
}

#[test]
fn stop_sequence_retains_matching_tokens() {
    let model = ScriptedModel::new(&[2, 3, 2, 4]);
    let control = GenerationControl { stop_token_sequences: vec![vec![3, 2]], ..Default::default() };
    let result = generate_causal_greedy_stream(
        &model, &limits(), &[1], &config(8), &control, &HostDevice,
        |_| ControlFlow::Continue(()),
    ).unwrap();
    assert_eq!(result.output.generated_token_ids, vec![2, 3, 2]);
    assert_eq!(result.finish_reason, GenerationFinishReason::StopSequence(0));
    assert!(!result.output.stopped_on_eos);
    assert_eq!(model.inputs.borrow().len(), 3);
}

#[test]
fn stop_sequence_does_not_match_across_prompt_boundary() {
    let model = ScriptedModel::new(&[2]);
    let control = GenerationControl { stop_token_sequences: vec![vec![1, 2]], ..Default::default() };
    let result = generate_causal_greedy_stream(
        &model, &limits(), &[1], &config(1), &control, &HostDevice,
        |_| ControlFlow::Continue(()),
    ).unwrap();
    assert_eq!(result.finish_reason, GenerationFinishReason::MaxNewTokens);
}

#[test]
fn eos_precedes_stop_sequence_length_and_callback_cancellation() {
    let model = ScriptedModel::new(&[2]);
    let control = GenerationControl { stop_token_sequences: vec![vec![2]], ..Default::default() };
    let generation = GreedyGenerationConfig { max_new_tokens: 1, eos_token_ids: vec![2] };
    let result = generate_causal_greedy_stream(
        &model, &limits(), &[1], &generation, &control, &HostDevice,
        |event| {
            assert_eq!(event.finish_reason, Some(GenerationFinishReason::EosToken(2)));
            ControlFlow::Break(())
        },
    ).unwrap();
    assert_eq!(result.finish_reason, GenerationFinishReason::EosToken(2));
    assert!(result.output.stopped_on_eos);
}

#[test]
fn stop_sequence_precedes_length_and_callback_cancellation() {
    let model = ScriptedModel::new(&[2]);
    let control = GenerationControl { stop_token_sequences: vec![vec![2]], ..Default::default() };
    let result = generate_causal_greedy_stream(
        &model, &limits(), &[1], &config(1), &control, &HostDevice,
        |_| ControlFlow::Break(()),
    ).unwrap();
    assert_eq!(result.finish_reason, GenerationFinishReason::StopSequence(0));
}

#[test]
fn zero_new_tokens_needs_no_model_execution() {
    let model = ScriptedModel::new(&[2]);
    let result = generate_causal_greedy_stream(
        &model, &limits(), &[1], &config(0), &GenerationControl::default(), &HostDevice,
        |_| panic!("zero-token request must not stream"),
    ).unwrap();
    assert_eq!(result.finish_reason, GenerationFinishReason::MaxNewTokens);
    assert_eq!(result.output.generated_token_ids, Vec::<i32>::new());
    assert_eq!(model.caches.get(), 0);
}

#[test]
fn invalid_controls_are_rejected_before_execution() {
    for patterns in [vec![vec![]], vec![vec![-1]], vec![vec![8]]] {
        let model = ScriptedModel::new(&[2]);
        let control = GenerationControl { stop_token_sequences: patterns, ..Default::default() };
        assert!(generate_causal_greedy_stream(
            &model, &limits(), &[1], &config(3), &control, &HostDevice,
            |_| ControlFlow::Continue(()),
        ).is_err());
        assert_eq!(model.caches.get(), 0);
    }
}

#[test]
fn invalid_prompt_eos_and_capacity_are_rejected() {
    let model = ScriptedModel::new(&[2]);
    for prompt in [vec![], vec![-1], vec![8]] {
        assert!(generate_causal_greedy(&model, &limits(), &prompt, &config(1), &HostDevice).is_err());
    }
    for eos_token_ids in [vec![-1], vec![8]] {
        let generation = GreedyGenerationConfig { max_new_tokens: 1, eos_token_ids };
        assert!(generate_causal_greedy(&model, &limits(), &[1], &generation, &HostDevice).is_err());
    }
    assert!(generate_causal_greedy(&model, &limits(), &[1], &config(usize::MAX), &HostDevice).is_err());
    assert!(generate_causal_greedy(&model, &limits(), &[1], &config(32), &HostDevice).is_err());
    assert_eq!(model.caches.get(), 0);
}

#[test]
fn malformed_model_output_and_selection_return_errors() {
    let mut model = ScriptedModel::new(&[2]);
    model.vocabulary = 7;
    assert!(generate_causal_greedy(&model, &limits(), &[1], &config(2), &HostDevice).is_err());
    model.vocabulary = 8;
    for token in [-1, 8] {
        model.forced_selection = Some(token);
        assert!(generate_causal_greedy(&model, &limits(), &[1], &config(2), &HostDevice).is_err());
    }
}

#[test]
fn fallible_model_errors_are_propagated() {
    let mut model = ScriptedModel::new(&[2]);
    model.fail_forward = true;
    let error = generate_causal_greedy(&model, &limits(), &[1], &config(2), &HostDevice).unwrap_err();
    assert_eq!(error.0, "scripted forward failure");
}

#[test]
fn existing_greedy_and_default_streaming_outputs_match() {
    let model = ScriptedModel::new(&[2, 3, 4]);
    let expected = generate_causal_greedy(&model, &limits(), &[1], &config(3), &HostDevice).unwrap();
    let actual = generate_causal_greedy_stream(
        &model, &limits(), &[1], &config(3), &GenerationControl::default(), &HostDevice,
        |_| ControlFlow::Continue(()),
    ).unwrap();
    assert_eq!(actual.output, expected);
}

#[test]
fn sampled_streaming_uses_the_same_stop_controls() {
    let model = ScriptedModel::new(&[2, 3, 4]);
    let generation = SamplingGenerationConfig {
        max_new_tokens: 8, eos_token_ids: vec![],
        sampling: SamplingConfig { seed: Some(42), ..Default::default() },
    };
    let control = GenerationControl { stop_token_sequences: vec![vec![2, 3]], ..Default::default() };
    let result = generate_causal_sampled_stream(
        &model, &limits(), &[1], &generation, &control, &HostDevice,
        |_| ControlFlow::Continue(()),
    ).unwrap();
    assert_eq!(result.output.generated_token_ids, vec![2, 3]);
    assert_eq!(result.finish_reason, GenerationFinishReason::StopSequence(0));
}

#[test]
fn existing_sampled_and_default_streaming_outputs_match() {
    let model = ScriptedModel::new(&[2, 3, 4]);
    let generation = SamplingGenerationConfig {
        max_new_tokens: 3, eos_token_ids: vec![],
        sampling: SamplingConfig { seed: Some(42), ..Default::default() },
    };
    let expected = generate_causal_sampled(&model, &limits(), &[1], &generation, &HostDevice).unwrap();
    let actual = generate_causal_sampled_stream(
        &model, &limits(), &[1], &generation, &GenerationControl::default(), &HostDevice,
        |_| ControlFlow::Continue(()),
    ).unwrap();
    assert_eq!(actual.output, expected);
}
