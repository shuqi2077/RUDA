use super::{
    CausalModel, CausalModelLimits, RudaPackedModel, GenerationError, GreedyGenerationConfig,
    TokenGenerationOutput, generate_with_selector,
};
use crate::{LlamaConfig, LlamaForCausalLm, PackedLlamaForCausalLm};
use ruda::runtime::server::ComputeServer;
use chacha20::ChaCha12Rng;
use ruda_core::rand::{RngExt, SeedableRng, get_seeded_rng};
use ruda_tensor::DeviceOps;
use ruda_tensor::api::backend::Backend;
use ruda_tensor::api::FloatDType;
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};

/// Temperature, top-k, then nucleus filtering for categorical token sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    /// Finite and strictly positive. Use the greedy API for deterministic argmax.
    pub temperature: f64,
    /// Zero disables top-k. Tokens tied at the kth logit are retained.
    pub top_k: usize,
    /// Probability mass to retain, in (0, 1]. One disables nucleus filtering.
    pub top_p: f64,
    /// Request-local random seed; None initializes from system entropy.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        }
    }
}

impl SamplingConfig {
    pub fn validate(&self) -> Result<(), GenerationError> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(GenerationError(
                "sampling temperature must be finite and positive".into(),
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(GenerationError("sampling top_p must be in (0, 1]".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplingGenerationConfig {
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<i32>,
    pub sampling: SamplingConfig,
}

impl Default for SamplingGenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            eos_token_ids: Vec::new(),
            sampling: SamplingConfig::default(),
        }
    }
}

/// A request-local sampler. Cloning preserves its exact current RNG state.
#[derive(Debug)]
pub struct TokenSampler {
    config: SamplingConfig,
    rng: ChaCha12Rng,
}

impl Clone for TokenSampler {
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            rng: ChaCha12Rng::deserialize_state(&self.rng.serialize_state()),
        }
    }
}

impl TokenSampler {
    pub fn new(config: SamplingConfig) -> Result<Self, GenerationError> {
        config.validate()?;
        let rng = config
            .seed
            .map(ChaCha12Rng::seed_from_u64)
            .unwrap_or_else(|| ChaCha12Rng::from_rng(&mut get_seeded_rng()));
        Ok(Self { config, rng })
    }

    /// Sample one vocabulary index. Negative infinity masks a token; NaN,
    /// positive infinity and a fully masked vocabulary return an error.
    pub fn sample(&mut self, logits: &[f32]) -> Result<i32, GenerationError> {
        let probabilities = filtered_probabilities(logits, self.config)?;
        let draw = self.rng.random::<f64>();
        let mut cumulative = 0.0;
        let mut last = None;
        for (token, probability) in probabilities.into_iter().enumerate() {
            if probability == 0.0 {
                continue;
            }
            last = Some(token);
            cumulative += probability;
            if draw < cumulative {
                return Ok(token as i32);
            }
        }
        // Summation rounding can leave the normalized CDF just below one.
        Ok(last.expect("validated nonempty sampling distribution") as i32)
    }
}

fn filtered_probabilities(
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

pub fn generate_sampled<B: Backend>(
    model: &LlamaForCausalLm<B>,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    device: &B::Device,
) -> Result<TokenGenerationOutput, GenerationError> {
    generate_sampled_impl(model, model_config, prompt_token_ids, generation, device)
}

pub fn generate_sampled_packed<B: Backend>(
    model: &PackedLlamaForCausalLm<B>,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    device: &B::Device,
) -> Result<TokenGenerationOutput, GenerationError> {
    generate_sampled_impl(model, model_config, prompt_token_ids, generation, device)
}

pub fn generate_sampled_packed_ruda<R, F, I, BT>(
    model: &PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    device: &R::Device,
) -> Result<TokenGenerationOutput, GenerationError>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    generate_sampled_impl(
        &RudaPackedModel(model),
        model_config,
        prompt_token_ids,
        generation,
        device,
    )
}

pub(super) fn generate_sampled_impl<B: Backend, M: CausalModel<B>>(
    model: &M,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    device: &B::Device,
) -> Result<TokenGenerationOutput, GenerationError> {
    generate_causal_sampled(model, &model_config.into(), prompt_token_ids, generation, device)
}

pub fn generate_causal_sampled<B: Backend, M: CausalModel<B>>(
    model: &M,
    model_limits: &CausalModelLimits,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    device: &B::Device,
) -> Result<TokenGenerationOutput, GenerationError> {
    let mut sampler = TokenSampler::new(generation.sampling)?;
    let limits = GreedyGenerationConfig {
        max_new_tokens: generation.max_new_tokens,
        eos_token_ids: generation.eos_token_ids.clone(),
    };
    generate_with_selector(
        model,
        model_limits,
        prompt_token_ids,
        &limits,
        device,
        |_, logits| {
            let [batch, sequence, vocabulary] = logits.dims();
            if batch != 1 || sequence == 0 || vocabulary == 0 {
                return Err(GenerationError(format!(
                    "expected non-empty batch-one logits, got [{batch}, {sequence}, {vocabulary}]"
                )));
            }
            let row = logits
                .slice([0..1, sequence - 1..sequence, 0..vocabulary])
                .cast(FloatDType::F32)
                .try_into_data()
                .map_err(|error| GenerationError(format!("cannot read sampling logits: {error}")))?
                .to_vec::<f32>()
                .map_err(|error| {
                    GenerationError(format!("cannot decode sampling logits: {error}"))
                })?;
            sampler.sample(&row)
        },
    )
}

#[cfg(test)]
mod tests;
