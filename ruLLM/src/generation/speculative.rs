use super::{
    CausalModel, CausalModelLimits, GenerationControl, GenerationError, GenerationEvent,
    GenerationFinishReason, GreedyGenerationConfig, SamplingConfig, StopSequenceMatcher,
    TokenGenerationOutput, read_last_logits, validate_generation,
};
use super::sampling::filtered_probabilities;
use chacha20::ChaCha12Rng;
use ruda_core::rand::{RngExt, SeedableRng, get_seeded_rng};
use ruda_tensor::api::{FloatDType, Int, Tensor, TensorData, backend::Backend};
use std::ops::ControlFlow;

/// An opt-in extension: ordinary cached generation implementations are unchanged.
pub trait SpeculativeModel<B: Backend>: CausalModel<B> {
    /// Fork ALL semantic state, including position, recurrent and convolution state.
    /// Future writes to either branch must not change the other branch.
    fn fork_cache(&self, cache: &Self::Cache) -> Self::Cache;

    /// Append the whole candidate block in one model forward, returning one logit
    /// row per input token. Row i predicts the token AFTER input i.
    fn try_forward_cached_block(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Result<Tensor<B, 3>, GenerationError>;
}

impl<B: Backend> SpeculativeModel<B> for crate::LlamaForCausalLm<B> {
    fn fork_cache(&self, cache: &Self::Cache) -> Self::Cache { cache.clone() }

    fn try_forward_cached_block(
        &self, tokens: Tensor<B, 2, Int>, cache: &mut Self::Cache,
    ) -> Result<Tensor<B, 3>, GenerationError> {
        Ok(self.forward_cached(tokens, cache))
    }
}

/// Batch-one speculation. Target and draft MUST use the same token-ID semantics;
/// equal vocabulary sizes alone do not establish tokenizer compatibility.
#[derive(Debug, Clone)]
pub struct SpeculativeGenerationConfig {
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<i32>,
    pub draft_tokens: usize,
    /// None selects greedy verification. Some applies the existing temperature,
    /// top-k and top-p policy to both models, with rejection-corrected sampling.
    pub sampling: Option<SamplingConfig>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    pub rounds: usize,
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
    pub rejected_blocks: usize,
    pub bonus_tokens: usize,
    pub target_block_forwards: usize,
    pub replayed_prefix_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeGenerationOutput {
    pub output: TokenGenerationOutput,
    pub finish_reason: GenerationFinishReason,
    pub stats: SpeculativeStats,
}

pub fn generate_causal_speculative<B, T, D>(
    target: &T, target_limits: &CausalModelLimits,
    draft: &D, draft_limits: &CausalModelLimits,
    prompt: &[i32], config: &SpeculativeGenerationConfig, device: &B::Device,
) -> Result<SpeculativeGenerationOutput, GenerationError>
where B: Backend, T: SpeculativeModel<B>, D: SpeculativeModel<B> {
    generate_causal_speculative_stream(
        target, target_limits, draft, draft_limits, prompt, config,
        &GenerationControl::default(), device, |_| ControlFlow::Continue(()),
    )
}

/// Emits only verified tokens. EOS, stop sequences, length and callback/cancel
/// ordering follow ordinary generation. A callback never sees a rejected draft.
pub fn generate_causal_speculative_stream<B, T, D>(
    target: &T, target_limits: &CausalModelLimits,
    draft: &D, draft_limits: &CausalModelLimits,
    prompt: &[i32], config: &SpeculativeGenerationConfig,
    control: &GenerationControl, device: &B::Device,
    mut on_token: impl FnMut(GenerationEvent<'_>) -> ControlFlow<()>,
) -> Result<SpeculativeGenerationOutput, GenerationError>
where B: Backend, T: SpeculativeModel<B>, D: SpeculativeModel<B> {
    let ordinary = GreedyGenerationConfig {
        max_new_tokens: config.max_new_tokens, eos_token_ids: config.eos_token_ids.clone(),
    };
    let eos = validate_generation(target_limits, prompt, &ordinary)?;
    validate_generation(draft_limits, prompt, &ordinary)?;
    if config.draft_tokens == 0 || target_limits.vocab_size != draft_limits.vocab_size {
        return Err(GenerationError("speculation requires a positive draft length and matching vocabularies".into()));
    }
    if let Some(sampling) = config.sampling { sampling.validate()?; }
    control.validate(target_limits.vocab_size)?;
    let mut result = SpeculativeGenerationOutput {
        output: TokenGenerationOutput {
            token_ids: prompt.to_vec(), generated_token_ids: Vec::new(), stopped_on_eos: false,
        },
        finish_reason: GenerationFinishReason::MaxNewTokens,
        stats: SpeculativeStats::default(),
    };
    if control.is_cancelled() { result.finish_reason = GenerationFinishReason::Cancelled; }
    if config.max_new_tokens == 0 || control.is_cancelled() { return Ok(result); }
    let mut selection = Selection::new(config.sampling);
    let vocabulary = target_limits.vocab_size;
    let mut matcher = StopSequenceMatcher::new(&control.stop_token_sequences);
    let mut target_cache = target.new_cache();
    let mut draft_cache = draft.new_cache();
    let mut target_next = next_row(target, prompt, &mut target_cache, vocabulary, device)?;
    let mut draft_next = next_row(draft, prompt, &mut draft_cache, vocabulary, device)?;

    'generation: while result.output.generated_token_ids.len() < config.max_new_tokens {
        if control.is_cancelled() {
            result.finish_reason = GenerationFinishReason::Cancelled;
            break;
        }
        let count = config.draft_tokens.min(config.max_new_tokens - result.output.generated_token_ids.len());
        let target_before = target.fork_cache(&target_cache);
        let draft_before = draft.fork_cache(&draft_cache);
        let mut candidates = Vec::with_capacity(count);
        let mut draft_probabilities = Vec::with_capacity(count);
        for _ in 0..count {
            if control.is_cancelled() {
                result.finish_reason = GenerationFinishReason::Cancelled;
                break 'generation;
            }
            let (token, distribution) = selection.propose(&draft_next)?;
            candidates.push(token);
            draft_probabilities.push(distribution);
            draft_next = next_row(draft, &[token], &mut draft_cache, vocabulary, device)?;
            if eos.contains(&token) { break; }
        }
        let logits = target.try_forward_cached_block(token_tensor(&candidates, device), &mut target_cache)?;
        let expected = [1, candidates.len(), vocabulary];
        if logits.dims() != expected {
            return Err(GenerationError(format!("expected speculative block logits {expected:?}, got {:?}", logits.dims())));
        }
        let rows = logits.cast(FloatDType::F32).try_into_data()
            .map_err(|e| GenerationError(format!("cannot read speculative logits: {e}")))?
            .to_vec::<f32>().map_err(|e| GenerationError(format!("cannot decode speculative logits: {e}")))?;
        result.stats.rounds += 1;
        result.stats.target_block_forwards += 1;
        result.stats.proposed_tokens += candidates.len();
        if control.is_cancelled() {
            result.finish_reason = GenerationFinishReason::Cancelled;
            break;
        }
        let mut committed = Vec::with_capacity(candidates.len() + 1);
        let mut rejected = false;
        for (index, &candidate) in candidates.iter().enumerate() {
            let p_logits = if index == 0 { &target_next[..] }
                else { &rows[(index - 1) * vocabulary..index * vocabulary] };
            let (token, accepted) = selection.verify(candidate, &draft_probabilities[index], p_logits)?;
            if accepted { result.stats.accepted_tokens += 1; }
            else { result.stats.rejected_blocks += 1; rejected = true; }
            committed.push(token);
            if emit(token, &mut result, config, &eos, &mut matcher, control, &mut on_token) {
                break 'generation;
            }
            if rejected { break; }
        }
        if rejected {
            // Hybrid recurrent state cannot be cropped like attention KV. Restore
            // the pre-block branch, then replay ONLY the committed prefix and its
            // correction token. Rejected candidates never survive into the next round.
            target_cache = target_before;
            draft_cache = draft_before;
            result.stats.replayed_prefix_tokens += committed.len();
            for token in committed {
                target_next = next_row(target, &[token], &mut target_cache, vocabulary, device)?;
                draft_next = next_row(draft, &[token], &mut draft_cache, vocabulary, device)?;
            }
        } else {
            // Both live branches include the entire accepted block. The final
            // verification row is the bonus distribution, not the last candidate's.
            let (bonus, _) = selection.propose(&rows[(candidates.len() - 1) * vocabulary..])?;
            result.stats.bonus_tokens += 1;
            if emit(bonus, &mut result, config, &eos, &mut matcher, control, &mut on_token) { break; }
            drop(target_before);
            drop(draft_before);
            target_next = next_row(target, &[bonus], &mut target_cache, vocabulary, device)?;
            draft_next = next_row(draft, &[bonus], &mut draft_cache, vocabulary, device)?;
        }
    }
    B::sync(device).map_err(|e| GenerationError(format!("speculative generation did not complete: {e}")))?;
    Ok(result)
}

fn token_tensor<B: Backend>(tokens: &[i32], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(tokens.to_vec(), [1, tokens.len()]), device)
}

fn next_row<B: Backend, M: CausalModel<B>>(
    model: &M, tokens: &[i32], cache: &mut M::Cache, vocabulary: usize, device: &B::Device,
) -> Result<Vec<f32>, GenerationError> {
    let logits = model.try_forward_cached_last(token_tensor(tokens, device), cache)?;
    if logits.dims() != [1, 1, vocabulary] {
        return Err(GenerationError(format!("expected last logits [1, 1, {vocabulary}], got {:?}", logits.dims())));
    }
    read_last_logits(logits)
}

fn emit(
    token: i32, result: &mut SpeculativeGenerationOutput, config: &SpeculativeGenerationConfig,
    eos: &std::collections::BTreeSet<i32>, matcher: &mut StopSequenceMatcher<'_>,
    control: &GenerationControl, on_token: &mut impl FnMut(GenerationEvent<'_>) -> ControlFlow<()>,
) -> bool {
    result.output.token_ids.push(token);
    result.output.generated_token_ids.push(token);
    let matched = matcher.push(token);
    let natural = if eos.contains(&token) {
        result.output.stopped_on_eos = true;
        Some(GenerationFinishReason::EosToken(token))
    } else if let Some(index) = matched { Some(GenerationFinishReason::StopSequence(index)) }
    else if result.output.generated_token_ids.len() == config.max_new_tokens { Some(GenerationFinishReason::MaxNewTokens) }
    else { None };
    let response = on_token(GenerationEvent {
        token_id: token, generated_token_ids: &result.output.generated_token_ids, finish_reason: natural,
    });
    if let Some(reason) = natural { result.finish_reason = reason; true }
    else if response.is_break() || control.is_cancelled() {
        result.finish_reason = GenerationFinishReason::Cancelled; true
    } else { false }
}

struct Selection { config: Option<SamplingConfig>, rng: ChaCha12Rng }

impl Selection {
    fn new(config: Option<SamplingConfig>) -> Self {
        let rng = config.and_then(|c| c.seed).map(ChaCha12Rng::seed_from_u64)
            .unwrap_or_else(|| ChaCha12Rng::from_rng(&mut get_seeded_rng()));
        Self { config, rng }
    }

    fn propose(&mut self, logits: &[f32]) -> Result<(i32, Vec<f64>), GenerationError> {
        if let Some(config) = self.config {
            let probabilities = filtered_probabilities(logits, config)?;
            Ok((draw(&probabilities, self.rng.random())?, probabilities))
        } else { Ok((argmax(logits)?, Vec::new())) }
    }

    fn verify(&mut self, candidate: i32, q: &[f64], logits: &[f32]) -> Result<(i32, bool), GenerationError> {
        if let Some(config) = self.config {
            let p = filtered_probabilities(logits, config)?;
            let index = candidate as usize;
            if self.rng.random::<f64>() < (p[index] / q[index]).min(1.0) { return Ok((candidate, true)); }
            let residual = p.iter().zip(q).map(|(p, q)| (p - q).max(0.0)).collect::<Vec<_>>();
            Ok((draw(&residual, self.rng.random())?, false))
        } else {
            let token = argmax(logits)?;
            Ok((token, token == candidate))
        }
    }
}

fn draw(probabilities: &[f64], uniform: f64) -> Result<i32, GenerationError> {
    let sum: f64 = probabilities.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(GenerationError("empty or nonfinite speculative sampling distribution".into()));
    }
    let threshold = uniform * sum;
    let mut cumulative = 0.0;
    let mut last = None;
    for (index, &probability) in probabilities.iter().enumerate() {
        if probability > 0.0 {
            last = Some(index as i32);
            cumulative += probability;
            if threshold < cumulative { return Ok(index as i32); }
        }
    }
    Ok(last.expect("nonempty validated distribution"))
}

fn argmax(logits: &[f32]) -> Result<i32, GenerationError> {
    let mut best = None;
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_nan() && best.is_none_or(|(_, maximum)| value > maximum) { best = Some((index, value)); }
    }
    best.map(|(index, _)| index as i32).ok_or_else(|| GenerationError("all vocabulary logits are NaN".into()))
}
