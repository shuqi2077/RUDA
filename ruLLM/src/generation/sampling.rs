use super::{
    CausalModel, CausalModelLimits, RudaPackedModel, GenerationError, GreedyGenerationConfig,
    TokenGenerationOutput, ControlledGenerationOutput, GenerationControl, GenerationEvent,
    generate_with_selector_stream, read_last_logits,
};
use crate::{LlamaConfig, LlamaForCausalLm, PackedLlamaForCausalLm};
use ruda::runtime::server::ComputeServer;
use chacha20::ChaCha12Rng;
use ruda_core::rand::{RngExt, SeedableRng, get_seeded_rng};
use ruda_tensor::DeviceOps;
use ruda_tensor::api::backend::Backend;
use std::ops::ControlFlow;
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};

mod distribution;
use distribution::SamplingWorkspace;

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
    workspace: SamplingWorkspace,
}

impl Clone for TokenSampler {
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            rng: ChaCha12Rng::deserialize_state(&self.rng.serialize_state()),
            // Scratch buffers are not semantic state. Avoid copying a full
            // vocabulary when the scheduler previews a request's next draw.
            workspace: SamplingWorkspace::default(),
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
        Ok(Self { config, rng, workspace: SamplingWorkspace::default() })
    }

    /// Release reusable vocabulary buffers without changing configuration or
    /// RNG state. Useful when retaining many idle samplers between requests.
    pub fn clear_workspace(&mut self) {
        self.workspace = SamplingWorkspace::default();
    }

    /// Sample one vocabulary index. Negative infinity masks a token; NaN,
    /// positive infinity and a fully masked vocabulary return an error.
    pub fn sample(&mut self, logits: &[f32]) -> Result<i32, GenerationError> {
        let probabilities = self.workspace.fill(logits, self.config)?;
        let draw = self.rng.random::<f64>();
        let mut cumulative = 0.0;
        let mut last = None;
        for (token, probability) in probabilities.iter().copied().enumerate() {
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

pub(super) fn filtered_probabilities(
    logits: &[f32],
    config: SamplingConfig,
) -> Result<Vec<f64>, GenerationError> {
    Ok(SamplingWorkspace::default().fill(logits, config)?.to_vec())
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
    Ok(generate_causal_sampled_stream(
        model, model_limits, prompt_token_ids, generation,
        &GenerationControl::default(), device, |_| ControlFlow::Continue(()),
    )?.output)
}

/// Sample with the same filtering/RNG semantics as `generate_causal_sampled`,
/// while observing token callbacks, stop sequences and cooperative cancellation.
pub fn generate_causal_sampled_stream<B: Backend, M: CausalModel<B>>(
    model: &M,
    model_limits: &CausalModelLimits,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    control: &GenerationControl,
    device: &B::Device,
    on_token: impl FnMut(GenerationEvent<'_>) -> ControlFlow<()>,
) -> Result<ControlledGenerationOutput, GenerationError> {
    let mut sampler = TokenSampler::new(generation.sampling)?;
    let limits = GreedyGenerationConfig {
        max_new_tokens: generation.max_new_tokens,
        eos_token_ids: generation.eos_token_ids.clone(),
    };
    generate_with_selector_stream(
        model, model_limits, prompt_token_ids, &limits, control, device,
        |_, logits| sampler.sample(&read_last_logits(logits)?), on_token,
    )
}

#[cfg(test)]
mod tests;
