use crate::{LlamaConfig, LlamaForCausalLm, LlamaKvCache, PackedLlamaForCausalLm};
use ruda_tensor::api::backend::Backend;
use ruda_tensor::api::{FloatDType, Int, Tensor, TensorData};
use ruda_tensor::DeviceOps;
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};
use ruda::runtime::server::ComputeServer;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

mod awq;
mod control;
use control::StopSequenceMatcher;
use std::ops::ControlFlow;
pub use control::{
    ControlledGenerationOutput, GenerationCancellation, GenerationControl,
    GenerationEvent, GenerationFinishReason,
};
mod sampling;
pub use awq::{generate_greedy_awq, generate_sampled_awq};
pub use sampling::{
    SamplingConfig, SamplingGenerationConfig, TokenSampler, generate_causal_sampled, generate_sampled,
    generate_sampled_packed, generate_sampled_packed_ruda, generate_causal_sampled_stream,
};

/// Deterministic autoregressive decoding options. Sampling is deliberately not
/// hidden behind this type: each step selects the exact maximum logit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreedyGenerationConfig {
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<i32>,
}

impl Default for GreedyGenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            eos_token_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenGenerationOutput {
    /// Prompt followed by every generated token.
    pub token_ids: Vec<i32>,
    /// Only the tokens produced by the decoder.
    pub generated_token_ids: Vec<i32>,
    pub stopped_on_eos: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationError(pub String);

impl Display for GenerationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for GenerationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalModelLimits {
    pub vocab_size: usize,
    pub max_sequence_length: usize,
}

impl From<&LlamaConfig> for CausalModelLimits {
    fn from(config: &LlamaConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            max_sequence_length: config.max_sequence_length,
        }
    }
}

/// Run a batch-one cached prefill/decode loop. Greedy selection stays on the
/// device; only the selected token ID is read back for each step.
pub fn generate_greedy<B: Backend>(
    model: &LlamaForCausalLm<B>,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &GreedyGenerationConfig,
    device: &B::Device,
) -> Result<TokenGenerationOutput, GenerationError> {
    generate_greedy_impl(model, model_config, prompt_token_ids, generation, device)
}

/// Run the same deterministic cached decoder using projection weights packed
/// for inference. Token selection and stopping semantics are identical to
/// [`generate_greedy`].
pub fn generate_greedy_packed<B: Backend>(
    model: &PackedLlamaForCausalLm<B>,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &GreedyGenerationConfig,
    device: &B::Device,
) -> Result<TokenGenerationOutput, GenerationError> {
    generate_greedy_impl(model, model_config, prompt_token_ids, generation, device)
}

/// Run packed inference with the dedicated Ruda RMSNorm and SwiGLU kernels.
pub fn generate_greedy_packed_ruda<R, F, I, BT>(
    model: &PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &GreedyGenerationConfig,
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
    generate_greedy_impl(
        &RudaPackedModel(model),
        model_config,
        prompt_token_ids,
        generation,
        device,
    )
}

struct RudaPackedModel<'a, R, F, I, BT>(&'a PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>)
where
    R: DeviceRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement;

impl<R, F, I, BT> CausalModel<DeviceBackend<R, F, I, BT>> for RudaPackedModel<'_, R, F, I, BT>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    type Cache = LlamaKvCache<DeviceBackend<R, F, I, BT>>;

    fn new_cache(&self) -> LlamaKvCache<DeviceBackend<R, F, I, BT>> {
        self.0.new_cache()
    }

    fn forward_cached_last(
        &self,
        tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>,
        cache: &mut LlamaKvCache<DeviceBackend<R, F, I, BT>>,
    ) -> Tensor<DeviceBackend<R, F, I, BT>, 3> {
        self.0.forward_cached_last_ruda(tokens, cache)
    }

    fn greedy_token(
        &self,
        logits: Tensor<DeviceBackend<R, F, I, BT>, 3>,
    ) -> Result<i32, GenerationError> {
        let [batch, sequence, vocabulary] = logits.dims();
        if batch != 1 || sequence == 0 || vocabulary == 0 {
            return Err(GenerationError(format!(
                "expected non-empty batch-one logits, got [{batch}, {sequence}, {vocabulary}]"
            )));
        }
        let indices = logits
            .slice([0..1, sequence - 1..sequence, 0..vocabulary])
            .argmax(2)
            .try_into_data()
            .map_err(|error| GenerationError(format!("cannot read generation argmax: {error}")))?
            .to_vec::<i32>()
            .map_err(|error| {
                GenerationError(format!("cannot decode generation argmax: {error}"))
            })?;
        indices
            .first()
            .copied()
            .ok_or_else(|| GenerationError("generation argmax was empty".into()))
    }
}

pub trait CausalModel<B: Backend> {
    type Cache;

    fn new_cache(&self) -> Self::Cache;

    fn forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Tensor<B, 3>;

    /// Fallible execution used by generation. Existing infallible implementations remain valid.
    fn try_forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Result<Tensor<B, 3>, GenerationError> {
        Ok(self.forward_cached_last(tokens, cache))
    }

    fn greedy_token(&self, logits: Tensor<B, 3>) -> Result<i32, GenerationError> {
        greedy_token(logits)
    }
}

impl<B: Backend> CausalModel<B> for LlamaForCausalLm<B> {
    type Cache = LlamaKvCache<B>;

    fn new_cache(&self) -> LlamaKvCache<B> {
        LlamaForCausalLm::new_cache(self)
    }

    fn forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut LlamaKvCache<B>,
    ) -> Tensor<B, 3> {
        LlamaForCausalLm::forward_cached_last(self, tokens, cache)
    }
}

impl<B: Backend> CausalModel<B> for PackedLlamaForCausalLm<B> {
    type Cache = LlamaKvCache<B>;

    fn new_cache(&self) -> LlamaKvCache<B> {
        PackedLlamaForCausalLm::new_cache(self)
    }

    fn forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut LlamaKvCache<B>,
    ) -> Tensor<B, 3> {
        PackedLlamaForCausalLm::forward_cached_last(self, tokens, cache)
    }
}

fn generate_greedy_impl<B: Backend, M: CausalModel<B>>(
    model: &M,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &GreedyGenerationConfig,
    device: &B::Device,
) -> Result<TokenGenerationOutput, GenerationError> {
    generate_causal_greedy(model, &model_config.into(), prompt_token_ids, generation, device)
}

pub fn generate_causal_greedy<B: Backend, M: CausalModel<B>>(
    model: &M,
    model_limits: &CausalModelLimits,
    prompt_token_ids: &[i32],
    generation: &GreedyGenerationConfig,
    device: &B::Device,
) -> Result<TokenGenerationOutput, GenerationError> {
    generate_with_selector(model, model_limits, prompt_token_ids, generation, device, |model, logits| {
        model.greedy_token(logits)
    })
}

/// Cached greedy generation with token-level streaming and request controls.
/// The callback runs synchronously once per selected token. Return
/// `ControlFlow::Break(())` to stop after retaining that token. Natural stopping
/// (EOS, then stop sequence, then length) takes precedence over cancellation.
pub fn generate_causal_greedy_stream<B: Backend, M: CausalModel<B>>(
    model: &M,
    model_limits: &CausalModelLimits,
    prompt_token_ids: &[i32],
    generation: &GreedyGenerationConfig,
    control: &GenerationControl,
    device: &B::Device,
    on_token: impl FnMut(GenerationEvent<'_>) -> ControlFlow<()>,
) -> Result<ControlledGenerationOutput, GenerationError> {
    generate_with_selector_stream(
        model, model_limits, prompt_token_ids, generation, control, device,
        |model, logits| model.greedy_token(logits), on_token,
    )
}

fn generate_with_selector<B: Backend, M: CausalModel<B>>(
    model: &M,
    model_config: &CausalModelLimits,
    prompt_token_ids: &[i32],
    generation: &GreedyGenerationConfig,
    device: &B::Device,
    select: impl FnMut(&M, Tensor<B, 3>) -> Result<i32, GenerationError>,
) -> Result<TokenGenerationOutput, GenerationError> {
    Ok(generate_with_selector_stream(
        model, model_config, prompt_token_ids, generation,
        &GenerationControl::default(), device, select,
        |_| ControlFlow::Continue(()),
    )?.output)
}

fn validate_generation(
    model_config: &CausalModelLimits,
    prompt_token_ids: &[i32],
    generation: &GreedyGenerationConfig,
) -> Result<BTreeSet<i32>, GenerationError> {
    if model_config.vocab_size == 0 || model_config.vocab_size - 1 > i32::MAX as usize {
        return Err(GenerationError("model requires a nonempty i32-indexed vocabulary".into()));
    }
    if prompt_token_ids.is_empty() {
        return Err(GenerationError(
            "generation requires at least one prompt token".into(),
        ));
    }
    for &token in prompt_token_ids {
        if token < 0 || token as usize >= model_config.vocab_size {
            return Err(GenerationError(format!(
                "prompt token {token} is outside vocabulary [0, {})",
                model_config.vocab_size
            )));
        }
    }
    let requested_length = prompt_token_ids
        .len()
        .checked_add(generation.max_new_tokens)
        .ok_or_else(|| GenerationError("generation length overflow".into()))?;
    if requested_length > model_config.max_sequence_length {
        return Err(GenerationError(format!(
            "prompt plus max_new_tokens ({requested_length}) exceeds model capacity {}",
            model_config.max_sequence_length
        )));
    }
    let eos = generation
        .eos_token_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if let Some(invalid) = eos
        .iter()
        .find(|&&token| token < 0 || token as usize >= model_config.vocab_size)
    {
        return Err(GenerationError(format!(
            "EOS token {invalid} is outside vocabulary [0, {})",
            model_config.vocab_size
        )));
    }

    Ok(eos)
}

fn generate_with_selector_stream<B: Backend, M: CausalModel<B>>(
    model: &M,
    model_config: &CausalModelLimits,
    prompt_token_ids: &[i32],
    generation: &GreedyGenerationConfig,
    control: &GenerationControl,
    device: &B::Device,
    mut select: impl FnMut(&M, Tensor<B, 3>) -> Result<i32, GenerationError>,
    mut on_token: impl FnMut(GenerationEvent<'_>) -> ControlFlow<()>,
) -> Result<ControlledGenerationOutput, GenerationError> {
    let eos = validate_generation(model_config, prompt_token_ids, generation)?;
    control.validate(model_config.vocab_size)?;
    let mut output = TokenGenerationOutput {
        token_ids: prompt_token_ids.to_vec(),
        generated_token_ids: Vec::new(),
        stopped_on_eos: false,
    };
    if generation.max_new_tokens == 0 || control.is_cancelled() {
        return Ok(ControlledGenerationOutput {
            output,
            finish_reason: if control.is_cancelled() {
                GenerationFinishReason::Cancelled
            } else {
                GenerationFinishReason::MaxNewTokens
            },
        });
    }
    // Bound eager allocation even when a caller supplies a very large limit.
    let initial_capacity = generation.max_new_tokens.min(1024);
    output.token_ids.reserve(initial_capacity);
    output.generated_token_ids.reserve(initial_capacity);
    let mut matcher = StopSequenceMatcher::new(&control.stop_token_sequences);
    let mut cache = model.new_cache();
    let prompt = Tensor::<B, 2, Int>::from_data(
        TensorData::new(prompt_token_ids.to_vec(), [1, prompt_token_ids.len()]), device,
    );
    let mut logits = model.try_forward_cached_last(prompt, &mut cache)?;
    let mut finish_reason = GenerationFinishReason::MaxNewTokens;

    for step in 0..generation.max_new_tokens {
        if control.is_cancelled() {
            finish_reason = GenerationFinishReason::Cancelled;
            break;
        }
        let [batch, sequence, vocabulary] = logits.dims();
        if batch != 1 || sequence == 0 || vocabulary != model_config.vocab_size {
            return Err(GenerationError(format!(
                "expected batch-one logits with vocabulary {}, got [{batch}, {sequence}, {vocabulary}]",
                model_config.vocab_size
            )));
        }
        let token = select(model, logits)?;
        if token < 0 || token as usize >= model_config.vocab_size {
            return Err(GenerationError(format!(
                "selected token {token} is outside vocabulary [0, {})", model_config.vocab_size
            )));
        }
        output.token_ids.push(token);
        output.generated_token_ids.push(token);
        let matched = matcher.push(token);
        let natural_stop = if eos.contains(&token) {
            output.stopped_on_eos = true;
            Some(GenerationFinishReason::EosToken(token))
        } else if let Some(index) = matched {
            Some(GenerationFinishReason::StopSequence(index))
        } else if step + 1 == generation.max_new_tokens {
            Some(GenerationFinishReason::MaxNewTokens)
        } else {
            None
        };
        let response = on_token(GenerationEvent {
            token_id: token,
            generated_token_ids: &output.generated_token_ids,
            finish_reason: natural_stop,
        });
        if let Some(reason) = natural_stop {
            finish_reason = reason;
            break;
        }
        if response.is_break() || control.is_cancelled() {
            finish_reason = GenerationFinishReason::Cancelled;
            break;
        }
        let input = Tensor::<B, 2, Int>::from_data([[token]], device);
        logits = model.try_forward_cached_last(input, &mut cache)?;
    }
    B::sync(device)
        .map_err(|error| GenerationError(format!("generation did not complete: {error}")))?;
    Ok(ControlledGenerationOutput { output, finish_reason })
}

fn read_last_logits<B: Backend>(logits: Tensor<B, 3>) -> Result<Vec<f32>, GenerationError> {
    let [batch, sequence, vocabulary] = logits.dims();
    if batch != 1 || sequence == 0 || vocabulary == 0 {
        return Err(GenerationError(format!(
            "expected non-empty batch-one logits, got [{batch}, {sequence}, {vocabulary}]"
        )));
    }
    logits.slice([0..1, sequence - 1..sequence, 0..vocabulary])
        .cast(FloatDType::F32)
        .try_into_data()
        .map_err(|error| GenerationError(format!("cannot read sampling logits: {error}")))?
        .to_vec::<f32>()
        .map_err(|error| GenerationError(format!("cannot decode sampling logits: {error}")))
}

fn greedy_token<B: Backend>(logits: Tensor<B, 3>) -> Result<i32, GenerationError> {
    let [batch, sequence, vocabulary] = logits.dims();
    if batch != 1 || sequence == 0 || vocabulary == 0 {
        return Err(GenerationError(format!(
            "expected non-empty batch-one logits, got [{batch}, {sequence}, {vocabulary}]"
        )));
    }
    let row = logits
        .slice([0..1, sequence - 1..sequence, 0..vocabulary])
        .cast(FloatDType::F32);
    let valid = row.clone().is_nan().bool_not();
    let maximum = row.clone().mask_fill(valid.clone().bool_not(), f32::NEG_INFINITY).max_dim(2);
    let candidates = row.clone().equal(maximum).bool_and(valid.clone())
        .int().cast(ruda_tensor::api::IntDType::I64);
    let empty = candidates.clone().max_dim(2).equal_elem(0);
    let index = candidates.mask_where(empty, valid.int().cast(ruda_tensor::api::IntDType::I64)).argmax(2);
    let invalid = row.gather(2, index.clone()).is_nan();
    let selected = index.mask_fill(invalid, -1)
        .try_into_data()
        .map_err(|error| GenerationError(format!("cannot read generation argmax: {error}")))?
        .to_vec::<i64>()
        .map_err(|error| GenerationError(format!("cannot decode generation argmax: {error}")))?;
    let index = *selected.first()
        .ok_or_else(|| GenerationError("generation argmax was empty".into()))?;
    if index == -1 {
        return Err(GenerationError("all vocabulary logits are NaN".into()));
    }
    i32::try_from(index)
        .map_err(|_| GenerationError("vocabulary index does not fit an i32 token id".into()))
}
