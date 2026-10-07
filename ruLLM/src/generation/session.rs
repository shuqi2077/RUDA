use super::{CausalModel,CausalModelLimits,ControlledGenerationOutput,GenerationControl,GenerationError,
    GenerationEvent,GenerationFinishReason,GreedyGenerationConfig,SamplingGenerationConfig,
    TokenGenerationOutput,TokenSampler,read_last_logits,validate_generation};
use super::control::StopSequenceMatcher;
use ruda_tensor::api::{Int,Tensor,TensorData,backend::Backend};
use std::{collections::BTreeSet,ops::ControlFlow};

mod record;
pub use record::{GenerationCacheContinuation,GenerationSessionRecord,GenerationSessionState};

/// Actual selected-token event from one incremental generation step.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct GenerationStep {
    /// The token retained in this session's generated suffix.
    pub token_id: i32,
    /// Actual number of selected new tokens, not number of cache input forwards.
    pub generated_tokens: usize,
    /// Natural termination known immediately after selecting this token.
    pub finish_reason: Option<GenerationFinishReason>,
}

/// One caller-driven cached request with native greedy or request-local sampling.
/// The model and controls are borrowed; cached tensors and output history are owned.
/// No architecture, positional scheme, BOS token or model-family rule is inferred.
pub struct CausalGenerationSession<'a,B: Backend,M: CausalModel<B>> {
    model: &'a M,
    device: B::Device,
    limits: CausalModelLimits,
    generation: GreedyGenerationConfig,
    control: &'a GenerationControl,
    matcher: StopSequenceMatcher<'a>,
    eos: BTreeSet<i32>,
    cache: M::Cache,
    logits: Option<Tensor<B,3>>,
    pending_input: Option<i32>,
    sampler: Option<TokenSampler>,
    prompt_length: usize,
    processed_tokens: usize,
    output: TokenGenerationOutput,
    finish_reason: Option<GenerationFinishReason>,
    fault: Option<GenerationError>,
}

impl<'a,B: Backend,M: CausalModel<B>> CausalGenerationSession<'a,B,M> {
    /// Prefill the caller's actual prompt once and prepare incremental greedy decoding.
    /// A zero generation limit or already-cancelled control executes no model forward.
    pub fn new_greedy(model: &'a M,limits: CausalModelLimits,prompt: &[i32],generation: GreedyGenerationConfig,
        control: &'a GenerationControl,device: &B::Device) -> Result<Self,GenerationError> {
        Self::new(model,limits,prompt,generation,None,control,device)
    }

    /// Prefill once with the existing temperature/top-k/nucleus sampling semantics.
    /// The session owns the sampler's exact evolving RNG state, not only its seed.
    pub fn new_sampled(model: &'a M,limits: CausalModelLimits,prompt: &[i32],generation: SamplingGenerationConfig,
        control: &'a GenerationControl,device: &B::Device) -> Result<Self,GenerationError> {
        let sampler = TokenSampler::new(generation.sampling)?;
        let generation = GreedyGenerationConfig {max_new_tokens:generation.max_new_tokens,eos_token_ids:generation.eos_token_ids};
        Self::new(model,limits,prompt,generation,Some(sampler),control,device)
    }

    fn new(model: &'a M,limits: CausalModelLimits,prompt: &[i32],generation: GreedyGenerationConfig,
        sampler: Option<TokenSampler>,control: &'a GenerationControl,device: &B::Device) -> Result<Self,GenerationError> {
        let eos = validate_generation(&limits,prompt,&generation)?;
        control.validate(limits.vocab_size)?;
        let finish_reason = if control.is_cancelled() {Some(GenerationFinishReason::Cancelled)}
            else if generation.max_new_tokens == 0 {Some(GenerationFinishReason::MaxNewTokens)} else {None};
        let mut session = Self {model,device:device.clone(),limits,generation,control,
            matcher:StopSequenceMatcher::new(&control.stop_token_sequences),eos,cache:model.new_cache(),
            logits:None,pending_input:None,sampler,prompt_length:prompt.len(),processed_tokens:0,
            output:TokenGenerationOutput {token_ids:prompt.to_vec(),generated_token_ids:Vec::new(),stopped_on_eos:false},
            finish_reason,fault:None};
        if session.finish_reason.is_none() {
            let input = Tensor::<B,2,Int>::from_data(TensorData::new(prompt.to_vec(),[1,prompt.len()]),device);
            let logits = session.model.try_forward_cached_last(input,&mut session.cache)?;
            session.check_logits(&logits)?;
            session.logits = Some(logits);
            session.processed_tokens = prompt.len();
        }
        Ok(session)
    }

    fn check_logits(&self,logits: &Tensor<B,3>) -> Result<(),GenerationError> {
        let [batch,sequence,vocabulary] = logits.dims();
        if batch != 1 || sequence == 0 || vocabulary != self.limits.vocab_size || logits.device() != self.device {
            return Err(GenerationError(format!("expected same-device batch-one logits with vocabulary {}, got [{batch},{sequence},{vocabulary}]",self.limits.vocab_size)));
        }
        Ok(())
    }

    /// Actual immutable output so far; the prompt and generated suffix stay separate.
    pub fn output(&self) -> &TokenGenerationOutput {&self.output}
    /// Exact natural or cooperative termination, if the request has ended.
    pub fn finish_reason(&self) -> Option<GenerationFinishReason> {self.finish_reason}
    /// Actual selected new-token count, including retained EOS/stop-pattern tokens.
    pub fn generated_tokens(&self) -> usize {self.output.generated_token_ids.len()}
    /// Number of tokens actually forwarded into cache, excluding a pending selection.
    pub fn processed_tokens(&self) -> usize {self.processed_tokens}
    /// Actual cache history; inspecting it does not consume or replay any input.
    pub fn cache(&self) -> &M::Cache {&self.cache}
    /// Original request-local sampler state, if this is a sampling request.
    pub fn sampler_state(&self) -> Option<super::sampling::TokenSamplerState> {self.sampler.as_ref().map(TokenSampler::to_state)}
    /// A forward/readback failure may leave model-specific cache state incomplete.
    /// It is retained for diagnosis, not silently retried or marked checkpointable.
    pub fn fault(&self) -> Option<&GenerationError> {self.fault.as_ref()}

    /// Select one actual new token. The previous selected token is forwarded only
    /// when this next step is requested; prompt/history is never replayed.
    /// EOS, then generated-only stop pattern, then length take precedence. Explicit
    /// cancellation is observed between steps and never interrupts a running kernel.
    pub fn step(&mut self) -> Result<Option<GenerationStep>,GenerationError> {
        if let Some(error) = &self.fault {return Err(error.clone());}
        let result = self.step_inner();
        if let Err(error) = &result {self.fault = Some(error.clone());}
        result
    }

    fn step_inner(&mut self) -> Result<Option<GenerationStep>,GenerationError> {
        if self.finish_reason.is_some() {return Ok(None);}
        if self.control.is_cancelled() {self.finish(GenerationFinishReason::Cancelled);return Ok(None);}
        if let Some(token) = self.pending_input {
            let input = Tensor::<B,2,Int>::from_data([[token]],&self.device);
            let logits = self.model.try_forward_cached_last(input,&mut self.cache)?;
            self.check_logits(&logits)?;
            self.logits = Some(logits);
            self.pending_input = None;
            self.processed_tokens = self.processed_tokens.checked_add(1)
                .ok_or_else(||GenerationError("generation cache input count overflow".into()))?;
        }
        let logits = self.logits.as_ref().ok_or_else(||GenerationError("generation has no pending logits or decode input".into()))?.clone();
        let token = if let Some(sampler) = &mut self.sampler {sampler.sample(&read_last_logits(logits)?)?}
            else {self.model.greedy_token(logits)?};
        if token < 0 || token as usize >= self.limits.vocab_size {
            return Err(GenerationError(format!("selected token {token} is outside the vocabulary")));
        }
        self.logits = None;
        self.output.token_ids.push(token);
        self.output.generated_token_ids.push(token);
        let matched = self.matcher.push(token);
        let reason = if self.eos.contains(&token) {
            self.output.stopped_on_eos = true;
            Some(GenerationFinishReason::EosToken(token))
        } else if let Some(index) = matched {Some(GenerationFinishReason::StopSequence(index))}
            else if self.generated_tokens() == self.generation.max_new_tokens {Some(GenerationFinishReason::MaxNewTokens)} else {None};
        if let Some(reason) = reason {self.finish(reason);} else {self.pending_input = Some(token);}
        Ok(Some(GenerationStep {token_id:token,generated_tokens:self.generated_tokens(),finish_reason:reason}))
    }

    fn finish(&mut self,reason: GenerationFinishReason) {
        self.finish_reason = Some(reason);
        self.pending_input = None;
        self.logits = None;
    }

    /// Run all remaining requested steps, with no interactive pauses or confirmations.
    /// Returns only after the backend's normal final synchronization completes.
    pub fn run(&mut self) -> Result<ControlledGenerationOutput,GenerationError> {
        self.run_stream(|_|ControlFlow::Continue(()))
    }

    /// Resume streaming from the current boundary; prior events are not emitted again.
    /// Callback Break retains the selected token and follows existing Cancelled semantics.
    /// Natural stopping still wins over callback cancellation on the same token.
    pub fn run_stream(&mut self,mut on_token: impl FnMut(GenerationEvent<'_>) -> ControlFlow<()>)
        -> Result<ControlledGenerationOutput,GenerationError> {
        while let Some(step) = self.step()? {
            let response = on_token(GenerationEvent {token_id:step.token_id,
                generated_token_ids:&self.output.generated_token_ids,finish_reason:step.finish_reason});
            if step.finish_reason.is_none() && (response.is_break() || self.control.is_cancelled()) {
                self.finish(GenerationFinishReason::Cancelled);
            }
        }
        if self.processed_tokens > 0 {
            if let Err(error) = B::sync(&self.device) {
                let error = GenerationError(format!("generation did not complete: {error}"));
                self.fault = Some(error.clone());
                return Err(error);
            }
        }
        Ok(ControlledGenerationOutput {output:self.output.clone(),
            finish_reason:self.finish_reason.expect("incremental generation reached a terminal state")})
    }
}
