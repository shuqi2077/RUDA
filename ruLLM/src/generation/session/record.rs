use super::{Backend,CausalGenerationSession,CausalModel,CausalModelLimits,GenerationControl,
    GenerationError,GenerationFinishReason,GreedyGenerationConfig,StopSequenceMatcher,TokenGenerationOutput,
    TokenSampler,validate_generation};
use crate::TokenSamplerState;
use ruda_model::record::{FullPrecisionSettings,PrecisionSettings,Record,Recorder,RecorderError};
use ruda_nn::modules::cache::{EncoderDecoderKvCache,EncoderDecoderKvCacheRecord,TransformerKvCache,TransformerKvCacheRecord};
use ruda_tensor::api::{DType,Tensor,TensorData};
use std::marker::PhantomData;

/// Exact native cache continuation used by a checkpointable generation session.
/// Position counts actual inputs consumed since the model's fresh cache. Captured
/// records must keep their retained values unchanged when later decoding appends.
pub trait GenerationCacheContinuation<B: Backend>: Sized {
    /// Actual native record of this model's cache, not reconstructed token history.
    type Record: Record<B>;
    /// Number of actual prompt/decode inputs committed to this cache.
    fn generation_position(&self) -> usize;
    /// Capture complete actual cache values with the caller's exact model identity.
    fn capture_generation(&self,model_id: &str,device: &B::Device) -> Result<Self::Record,RecorderError>;
    /// Restore exact history against a fresh compatible model cache and input count.
    fn restore_generation(record: Self::Record,model_id: &str,template: &Self,position: usize,device: &B::Device)
        -> Result<Self,RecorderError>;
}

impl<B: Backend> GenerationCacheContinuation<B> for TransformerKvCache<B> {
    type Record = TransformerKvCacheRecord<B>;
    fn generation_position(&self) -> usize {self.position()}
    fn capture_generation(&self,model_id: &str,device: &B::Device) -> Result<Self::Record,RecorderError> {
        let item = self.record(model_id)?.into_item::<FullPrecisionSettings>();
        // Native cache from_item stores raw data and performs no tensor allocation.
        Ok(Self::Record::from_item::<FullPrecisionSettings>(item,device))
    }
    fn restore_generation(record: Self::Record,model_id: &str,template: &Self,position: usize,device: &B::Device)
        -> Result<Self,RecorderError> {
        let cache = record.restore(model_id,template.layers().len(),device)?;
        if cache.position() != position {return Err(invalid("decoder cache and request input positions differ"));}
        Ok(cache)
    }
}

impl<B: Backend> GenerationCacheContinuation<B> for EncoderDecoderKvCache<B> {
    type Record = EncoderDecoderKvCacheRecord<B>;
    fn generation_position(&self) -> usize {self.position()}
    fn capture_generation(&self,model_id: &str,device: &B::Device) -> Result<Self::Record,RecorderError> {
        let item = self.record(model_id)?.into_item::<FullPrecisionSettings>();
        Ok(Self::Record::from_item::<FullPrecisionSettings>(item,device))
    }
    fn restore_generation(record: Self::Record,model_id: &str,template: &Self,position: usize,device: &B::Device)
        -> Result<Self,RecorderError> {
        let cache = record.restore(model_id,template.decoder().layers().len(),device)?;
        if cache.position() != position {return Err(invalid("paired cache and request input positions differ"));}
        Ok(cache)
    }
}

/// Serialized request state at an actual completed token-step boundary.
/// Stop matcher prefixes are rebuilt from saved generated tokens, not the prompt.
#[derive(Clone,Debug,serde::Serialize,serde::Deserialize)]
pub struct GenerationSessionState {
    limits: CausalModelLimits,
    generation: GreedyGenerationConfig,
    stop_sequences: Vec<Vec<i32>>,
    prompt_length: usize,
    processed_tokens: usize,
    output: TokenGenerationOutput,
    pending_input: Option<i32>,
    sampler: Option<TokenSamplerState>,
    finish_reason: Option<GenerationFinishReason>,
}

impl GenerationSessionState {
    /// Actual output retained by this record, including the original prompt.
    pub fn output(&self) -> &TokenGenerationOutput {&self.output}
    /// Actual inputs consumed by the associated cache.
    pub fn processed_tokens(&self) -> usize {self.processed_tokens}
    /// Saved terminal state; None means another generation step remains.
    pub fn finish_reason(&self) -> Option<GenerationFinishReason> {self.finish_reason}
}

/// Original model cache, exact pending logits, token history and sampler together.
/// No model parameter copy or automatic model/source identity hash is created.
pub struct GenerationSessionRecord<B: Backend,R: Record<B>> {
    version: u32,
    model_id: String,
    state: GenerationSessionState,
    cache: R,
    logits: Option<TensorData>,
    backend: PhantomData<B>,
}

impl<B: Backend,R: Record<B>> Record<B> for GenerationSessionRecord<B,R> {
    type Item<S: PrecisionSettings> = (u32,String,GenerationSessionState,R::Item<S>,Option<TensorData>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.model_id,self.state,self.cache.into_item::<S>(),self.logits)
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,device: &B::Device) -> Self {
        Self {version:item.0,model_id:item.1,state:item.2,cache:R::from_item::<S>(item.3,device),logits:item.4,backend:PhantomData}
    }
}

fn invalid(reason: &str) -> RecorderError {RecorderError::Unknown(format!("invalid generation continuation: {reason}"))}

fn check_logits(data: &TensorData,vocabulary: usize) -> Result<(),RecorderError> {
    if data.rank() != 3 || data.shape[0] != 1 || data.shape[1] == 0 || data.shape[2] != vocabulary
        || !matches!(data.dtype,DType::F16|DType::BF16|DType::F32|DType::Flex32|DType::F64) {
        return Err(invalid("pending logits do not have original native batch-one vocabulary geometry"));
    }
    let bytes = data.shape.iter().try_fold(1usize,|count,&dimension|count.checked_mul(dimension))
        .and_then(|count|count.checked_mul(data.dtype.size())).ok_or_else(||invalid("pending logits byte count overflow"))?;
    if data.as_bytes().len() != bytes {return Err(invalid("pending logits byte count differs from native storage"));}
    Ok(())
}

impl<B: Backend,R: Record<B>> GenerationSessionRecord<B,R> {
    /// Inspect actual saved continuation state without restoring GPU tensors.
    pub fn state(&self) -> &GenerationSessionState {&self.state}
    /// Save one native record containing cache, RNG, inputs and pending output together.
    pub fn save<T: Recorder<B>>(self,recorder: &T,args: T::RecordArgs) -> Result<T::RecordOutput,RecorderError> {recorder.record(self,args)}
    /// Load a native record; original storage and model identity are checked in restore.
    pub fn load<T: Recorder<B>>(recorder: &T,args: T::LoadArgs,device: &B::Device) -> Result<Self,RecorderError> {recorder.load(args,device)}

    /// Resume actual cached generation, without prompt replay, repeated callbacks,
    /// a fresh random seed or changed generation limits. model_id identifies the
    /// original frozen weights/configuration/adapters and any model-owned source.
    /// Matching controls supply new live cancellation handles; an already saved
    /// terminal cancellation is retained, not silently resumed.
    pub fn restore<'a,M>(self,model: &'a M,limits: CausalModelLimits,model_id: &str,
        control: &'a GenerationControl,device: &B::Device) -> Result<CausalGenerationSession<'a,B,M>,RecorderError>
        where M: CausalModel<B>,M::Cache: GenerationCacheContinuation<B,Record=R> {
        if self.version != 1 || model_id.is_empty() || self.model_id != model_id || self.state.limits != limits {
            return Err(invalid("record version, exact model identity or actual limits differ"));
        }
        let state = self.state;
        let count = state.output.generated_token_ids.len();
        if state.prompt_length == 0 || state.prompt_length.checked_add(count) != Some(state.output.token_ids.len())
            || count > state.generation.max_new_tokens || state.output.token_ids[state.prompt_length..] != state.output.generated_token_ids {
            return Err(invalid("saved prompt, suffix or generated count differs"));
        }
        let eos = validate_generation(&limits,&state.output.token_ids[..state.prompt_length],&state.generation)
            .map_err(|error|invalid(&error.to_string()))?;
        control.validate(limits.vocab_size).map_err(|error|invalid(&error.to_string()))?;
        if state.stop_sequences != control.stop_token_sequences {return Err(invalid("generated-only stopping patterns differ"));}
        let mut matcher = StopSequenceMatcher::new(&control.stop_token_sequences);
        let mut natural = None;
        for (index,&token) in state.output.generated_token_ids.iter().enumerate() {
            if natural.is_some() || token < 0 || token as usize >= limits.vocab_size {
                return Err(invalid("suffix contains invalid tokens or tokens after natural termination"));
            }
            let matched = matcher.push(token);
            natural = if eos.contains(&token) {Some(GenerationFinishReason::EosToken(token))}
                else if let Some(index) = matched {Some(GenerationFinishReason::StopSequence(index))}
                else if index+1 == state.generation.max_new_tokens {Some(GenerationFinishReason::MaxNewTokens)} else {None};
        }
        let reason_valid = match state.finish_reason {
            Some(GenerationFinishReason::Cancelled) => natural.is_none(),
            Some(GenerationFinishReason::MaxNewTokens) if count == 0 => state.generation.max_new_tokens == 0,
            reason => reason == natural && (count > 0 || state.generation.max_new_tokens > 0),
        };
        if !reason_valid || state.output.stopped_on_eos != matches!(state.finish_reason,Some(GenerationFinishReason::EosToken(_))) {
            return Err(invalid("saved finish reason does not match actual generated history"));
        }
        let consumed = if count == 0 {state.prompt_length} else {state.prompt_length+count-1};
        if state.finish_reason.is_none() {
            if count == 0 {
                if state.processed_tokens != consumed || state.pending_input.is_some() || self.logits.is_none() {
                    return Err(invalid("prefilled request is missing exact pending logits"));
                }
            } else if state.processed_tokens != consumed || state.pending_input != state.output.generated_token_ids.last().copied() || self.logits.is_some() {
                return Err(invalid("decode input and actual consumed prefix differ"));
            }
        } else if state.pending_input.is_some() || self.logits.is_some()
            || (state.processed_tokens != consumed && !(count == 0 && state.processed_tokens == 0)) {
            return Err(invalid("terminal request contains uncommitted inputs or inconsistent consumed history"));
        }
        if count == 0 && state.generation.max_new_tokens == 0 && state.processed_tokens != 0 {
            return Err(invalid("zero-limit request cannot contain a forwarded prompt"));
        }
        if let Some(logits) = &self.logits {check_logits(logits,limits.vocab_size)?;}
        let sampler = state.sampler.map(TokenSampler::from_state).transpose().map_err(|error|invalid(&error.to_string()))?;
        let template = model.new_cache();
        if template.generation_position() != 0 {return Err(invalid("fresh generation cache must start at input position zero"));}
        let cache = M::Cache::restore_generation(self.cache,model_id,&template,state.processed_tokens,device)?;
        let logits = self.logits.map(|data| {let dtype = data.dtype;Tensor::from_data(data,(device,dtype))});
        Ok(CausalGenerationSession {model,device:device.clone(),limits,generation:state.generation,control,matcher,eos,cache,logits,
            pending_input:state.pending_input,sampler,prompt_length:state.prompt_length,processed_tokens:state.processed_tokens,
            output:state.output,finish_reason:state.finish_reason,fault:None})
    }
}

impl<'a,B: Backend,M: CausalModel<B>> CausalGenerationSession<'a,B,M>
    where M::Cache: GenerationCacheContinuation<B> {
    /// Snapshot the actual completed request, including model-specific cache and RNG.
    /// Native pending logits keep exact storage independently of recorder precision.
    /// A failed forward/readback is not checkpointable; recover from an earlier record.
    pub fn record(&mut self,model_id: &str) -> Result<GenerationSessionRecord<B,<M::Cache as GenerationCacheContinuation<B>>::Record>,RecorderError> {
        if model_id.is_empty() || self.fault.is_some() {return Err(invalid("exact identity and a nonfaulted token boundary are required"));}
        if self.cache.generation_position() != self.processed_tokens {return Err(invalid("cache is not at the request's completed input position"));}
        if self.processed_tokens > 0 {
            if let Err(error) = B::sync(&self.device) {
                let error = GenerationError(format!("generation checkpoint did not complete: {error}"));
                self.fault = Some(error.clone());
                return Err(invalid(&error.to_string()));
            }
        }
        // Blocking readback follows native record APIs; no model forward or RNG draw occurs.
        let logits = match &self.logits {
            Some(logits) => {
                let data = logits.clone().try_into_data().map_err(|error|invalid(&error.to_string()))?;
                check_logits(&data,self.limits.vocab_size)?;
                Some(data)
            }
            None => None,
        };
        let cache = self.cache.capture_generation(model_id,&self.device)?;
        let state = GenerationSessionState {limits:self.limits,generation:self.generation.clone(),
            stop_sequences:self.control.stop_token_sequences.clone(),prompt_length:self.prompt_length,
            processed_tokens:self.processed_tokens,output:self.output.clone(),pending_input:self.pending_input,
            sampler:self.sampler_state(),finish_reason:self.finish_reason};
        Ok(GenerationSessionRecord {version:1,model_id:model_id.into(),state,cache,logits,backend:PhantomData})
    }
}
