use super::*;
use crate::{CausalModelLimits, GenerationError, SamplingGenerationConfig};
use ruda_tensor::api::{FloatDType, Tensor, backend::Backend};

/// A real batched model adapter. Cache forks must isolate subsequent writes.
/// The model consumes scheduled rows together, not by looping over model forwards.
pub trait DeviceBatchModel<B: Backend> {
    type Cache;
    fn batch_limits(&self) -> CausalModelLimits;
    fn new_batch_cache(&self, config: PagedKvCacheConfig) -> Self::Cache;
    fn fork_batch_cache(&self, cache: &Self::Cache) -> Self::Cache;
    fn forward_batch(
        &self,
        batch: &ScheduledBatch,
        cache: &mut Self::Cache,
        device: &B::Device,
    ) -> Result<Tensor<B, 3>, GenerationError>;
    /// Called only after device completion, when finished/cancelled pages can be released.
    fn retain_batch_cache(&self, cache: &mut Self::Cache, scheduler: &ContinuousBatchScheduler);
}

#[derive(Debug, Clone)]
pub struct ExecutedBatch {
    pub batch: ScheduledBatch,
    pub generated_token_ids: Vec<i32>,
}

/// Synchronous batch boundaries with asynchronous GPU work inside each forward.
/// The model is borrowed once; weights are shared across all resident requests.
pub struct DeviceBatchExecutor<'a, B: Backend, M: DeviceBatchModel<B>> {
    model: &'a M,
    device: B::Device,
    scheduler: ContinuousBatchScheduler,
    cache: M::Cache,
}

impl<'a, B: Backend, M: DeviceBatchModel<B>> DeviceBatchExecutor<'a, B, M> {
    pub fn new(
        model: &'a M,
        device: B::Device,
        config: ContinuousBatchConfig,
        kv: PagedKvCacheConfig,
        options: ContinuousBatchOptions,
    ) -> Result<Self, ContinuousBatchError> {
        if kv.max_sequence_length > model.batch_limits().max_sequence_length {
            return Err(ContinuousBatchError(
                "KV sequence limit exceeds model capacity".into(),
            ));
        }
        let scheduler = ContinuousBatchScheduler::with_options(config, kv, options)?;
        Ok(Self {
            model,
            device,
            scheduler,
            cache: model.new_batch_cache(kv),
        })
    }

    pub fn submit(
        &mut self,
        prompt: Vec<i32>,
        generation: GreedyGenerationConfig,
    ) -> Result<RequestId, ContinuousBatchError> {
        self.validate(&prompt, &generation)?;
        self.scheduler.submit(prompt, generation)
    }

    pub fn submit_sampled(
        &mut self,
        prompt: Vec<i32>,
        generation: SamplingGenerationConfig,
    ) -> Result<RequestId, ContinuousBatchError> {
        self.validate(
            &prompt,
            &GreedyGenerationConfig {
                max_new_tokens: generation.max_new_tokens,
                eos_token_ids: generation.eos_token_ids.clone(),
            },
        )?;
        self.scheduler.submit_sampled(prompt, generation)
    }

    fn validate(
        &self,
        prompt: &[i32],
        generation: &GreedyGenerationConfig,
    ) -> Result<(), ContinuousBatchError> {
        crate::generation::validate_generation(&self.model.batch_limits(), prompt, generation)
            .map(|_| ())
            .map_err(|e| ContinuousBatchError(e.to_string()))
    }

    pub fn step(&mut self) -> Result<Option<ExecutedBatch>, ContinuousBatchError> {
        let Some(batch) = self.scheduler.schedule()? else {
            return Ok(None);
        };
        let mut working = self.model.fork_batch_cache(&self.cache);
        let vocabulary = self.model.batch_limits().vocab_size;
        let result = self
            .model
            .forward_batch(&batch, &mut working, &self.device)
            .and_then(|logits| {
                if logits.dims() != [batch.batch_size(), 1, vocabulary] {
                    return Err(GenerationError("device batch logit shape mismatch".into()));
                }
                logits
                    .cast(FloatDType::F32)
                    .try_into_data()
                    .map_err(|e| GenerationError(e.to_string()))?
                    .to_vec::<f32>()
                    .map_err(|e| GenerationError(e.to_string()))
            });
        // An error/panic is not evidence of device completion. A failed sync
        // leaves this batch in flight; its pages cannot be cancelled or reused.
        B::sync(&self.device)
            .map_err(|e| ContinuousBatchError(format!("batch completion: {e}")))?;
        let selection = result
            .map_err(|e| ContinuousBatchError(e.to_string()))
            .and_then(|values| {
                self.scheduler
                    .select_batch_tokens(batch.id, &values, vocabulary)
            });
        let selection = match selection {
            Ok(selection) => selection,
            Err(error) => {
                self.scheduler.fail_batch(batch.id)?;
                return Err(error);
            }
        };
        let generated_token_ids = self.scheduler.complete_selected_batch(selection)?;
        self.model.retain_batch_cache(&mut working, &self.scheduler);
        self.cache = working;
        Ok(Some(ExecutedBatch {
            batch,
            generated_token_ids,
        }))
    }

    pub fn cancel(
        &mut self,
        id: RequestId,
    ) -> Result<Option<CancelledGeneration>, ContinuousBatchError> {
        let cancelled = self.scheduler.cancel(id)?;
        self.model
            .retain_batch_cache(&mut self.cache, &self.scheduler);
        Ok(cancelled)
    }
    pub fn pop_finished(&mut self) -> Option<FinishedGeneration> {
        self.scheduler.pop_finished()
    }
    pub fn is_idle(&self) -> bool {
        self.scheduler.is_idle()
    }
    pub fn snapshot(&self) -> ContinuousBatchSnapshot {
        self.scheduler.snapshot()
    }
    pub fn cache(&self) -> &M::Cache {
        &self.cache
    }
}
