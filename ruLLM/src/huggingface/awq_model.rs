use super::*;
use crate::{
    AwqLlamaForCausalLm, SamplingGenerationConfig, generate_greedy_awq, generate_sampled_awq,
};
use ruda_tensor::DeviceOps;
use ruda_tensor_device::DeviceRuntime;

#[cfg(all(test, feature = "nvidia"))]
mod tests;

#[derive(Debug)]
pub struct LoadedHuggingFaceAwqQwen2<R: DeviceRuntime>
where
    R::Device: DeviceOps,
{
    pub model: AwqLlamaForCausalLm<R>,
    pub config: LlamaConfig,
    pub default_eos_token_ids: Vec<i32>,
    pub report: HuggingFaceLoadReport,
}

pub fn load_huggingface_awq_qwen2<R: DeviceRuntime>(
    model_directory: impl AsRef<Path>,
    device: &R::Device,
) -> Result<LoadedHuggingFaceAwqQwen2<R>, HuggingFaceLoadError>
where
    R::Device: DeviceOps,
{
    let model_directory = canonical_directory(model_directory.as_ref())?;
    let config_path = model_directory.join(CONFIG_FILE);
    let source: HuggingFaceLlamaConfig = read_json(&config_path)?;
    let (config, tied_word_embeddings, config_eos_token_ids) = convert_qwen2_config(source)?;
    let default_eos_token_ids =
        load_generation_eos_token_ids(&model_directory, config.vocab_size, config_eos_token_ids)?;
    let checkpoint = AwqCheckpoint::open(&model_directory)?;
    let (model, applied_tensors) = AwqLlamaForCausalLm::<R>::load_checkpoint(
        &checkpoint,
        &config,
        true,
        tied_word_embeddings,
        device,
    )?;
    Ok(LoadedHuggingFaceAwqQwen2 {
        model,
        config,
        default_eos_token_ids,
        report: HuggingFaceLoadReport {
            config_path,
            weight_files: checkpoint.weight_files().to_vec(),
            applied_tensors,
            tied_word_embeddings,
        },
    })
}

impl<R: DeviceRuntime> LoadedHuggingFaceAwqQwen2<R>
where
    R::Device: DeviceOps,
{
    pub fn generate_tokens(
        &self,
        prompt_token_ids: &[i32],
        mut generation: GreedyGenerationConfig,
        device: &R::Device,
    ) -> Result<TokenGenerationOutput, GenerationError> {
        if generation.eos_token_ids.is_empty() {
            generation.eos_token_ids = self.default_eos_token_ids.clone();
        }
        generate_greedy_awq(
            &self.model,
            &self.config,
            prompt_token_ids,
            &generation,
            device,
        )
    }

    pub fn generate_tokens_sampled(
        &self,
        prompt_token_ids: &[i32],
        mut generation: SamplingGenerationConfig,
        device: &R::Device,
    ) -> Result<TokenGenerationOutput, GenerationError> {
        if generation.eos_token_ids.is_empty() {
            generation.eos_token_ids = self.default_eos_token_ids.clone();
        }
        generate_sampled_awq(
            &self.model,
            &self.config,
            prompt_token_ids,
            &generation,
            device,
        )
    }
}
