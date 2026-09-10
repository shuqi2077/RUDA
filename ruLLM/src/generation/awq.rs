use super::*;
use crate::llama::awq::{AwqBackend, AwqLlamaForCausalLm};

impl<R: DeviceRuntime> CausalModel<AwqBackend<R>> for AwqLlamaForCausalLm<R>
where
    R::Device: DeviceOps,
{
    type Cache = LlamaKvCache<AwqBackend<R>>;

    fn new_cache(&self) -> LlamaKvCache<AwqBackend<R>> {
        AwqLlamaForCausalLm::new_cache(self)
    }

    fn forward_cached_last(
        &self,
        tokens: Tensor<AwqBackend<R>, 2, Int>,
        cache: &mut LlamaKvCache<AwqBackend<R>>,
    ) -> Tensor<AwqBackend<R>, 3> {
        AwqLlamaForCausalLm::forward_cached_last(self, tokens, cache)
    }
}

pub fn generate_greedy_awq<R: DeviceRuntime>(
    model: &AwqLlamaForCausalLm<R>,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &GreedyGenerationConfig,
    device: &R::Device,
) -> Result<TokenGenerationOutput, GenerationError>
where
    R::Device: DeviceOps,
{
    generate_greedy_impl(model, model_config, prompt_token_ids, generation, device)
}

pub fn generate_sampled_awq<R: DeviceRuntime>(
    model: &AwqLlamaForCausalLm<R>,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    device: &R::Device,
) -> Result<TokenGenerationOutput, GenerationError>
where
    R::Device: DeviceOps,
{
    sampling::generate_sampled_impl(model, model_config, prompt_token_ids, generation, device)
}
