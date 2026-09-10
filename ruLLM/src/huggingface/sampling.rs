use super::{
    GeneratedText, HuggingFaceLlamaPipeline, HuggingFaceLoadError, HuggingFaceQwen2Pipeline,
    LoadedHuggingFaceLlama, LoadedHuggingFaceQwen2,
};
use crate::{GenerationError, SamplingGenerationConfig, TokenGenerationOutput, generate_sampled};
use ruda_tensor::api::backend::Backend;

impl<B: Backend> LoadedHuggingFaceLlama<B> {
    /// Sample with request-local RNG state and inherit the model's EOS IDs when omitted.
    pub fn generate_tokens_sampled(
        &self,
        prompt_token_ids: &[i32],
        mut generation: SamplingGenerationConfig,
        device: &B::Device,
    ) -> Result<TokenGenerationOutput, GenerationError> {
        if generation.eos_token_ids.is_empty() {
            generation.eos_token_ids = self.default_eos_token_ids.clone();
        }
        generate_sampled(
            &self.model,
            &self.config,
            prompt_token_ids,
            &generation,
            device,
        )
    }
}

impl<B: Backend> LoadedHuggingFaceQwen2<B> {
    /// Sample with request-local RNG state and inherit the model's EOS IDs when omitted.
    pub fn generate_tokens_sampled(
        &self,
        prompt_token_ids: &[i32],
        mut generation: SamplingGenerationConfig,
        device: &B::Device,
    ) -> Result<TokenGenerationOutput, GenerationError> {
        if generation.eos_token_ids.is_empty() {
            generation.eos_token_ids = self.default_eos_token_ids.clone();
        }
        generate_sampled(
            &self.model,
            &self.config,
            prompt_token_ids,
            &generation,
            device,
        )
    }
}

impl<B: Backend> HuggingFaceLlamaPipeline<B> {
    /// Tokenize, run cached sampled decoding, and decode the generated text.
    pub fn generate_text_sampled(
        &self,
        prompt: &str,
        generation: SamplingGenerationConfig,
        add_special_tokens: bool,
        skip_special_tokens: bool,
        device: &B::Device,
    ) -> Result<GeneratedText, HuggingFaceLoadError> {
        let prompt_ids = self.encode(prompt, add_special_tokens)?;
        let output = self
            .loaded
            .generate_tokens_sampled(&prompt_ids, generation, device)
            .map_err(|error| HuggingFaceLoadError(error.to_string()))?;
        Ok(GeneratedText {
            text: self.decode(&output.token_ids, skip_special_tokens)?,
            generated_text: self.decode(&output.generated_token_ids, skip_special_tokens)?,
            token_ids: output.token_ids,
            generated_token_ids: output.generated_token_ids,
            stopped_on_eos: output.stopped_on_eos,
        })
    }
}

impl<B: Backend> HuggingFaceQwen2Pipeline<B> {
    /// Tokenize, run cached sampled decoding, and decode the generated text.
    pub fn generate_text_sampled(
        &self,
        prompt: &str,
        generation: SamplingGenerationConfig,
        add_special_tokens: bool,
        skip_special_tokens: bool,
        device: &B::Device,
    ) -> Result<GeneratedText, HuggingFaceLoadError> {
        let prompt_ids = self.encode(prompt, add_special_tokens)?;
        let output = self
            .loaded
            .generate_tokens_sampled(&prompt_ids, generation, device)
            .map_err(|error| HuggingFaceLoadError(error.to_string()))?;
        Ok(GeneratedText {
            text: self.decode(&output.token_ids, skip_special_tokens)?,
            generated_text: self.decode(&output.generated_token_ids, skip_special_tokens)?,
            token_ids: output.token_ids,
            generated_token_ids: output.generated_token_ids,
            stopped_on_eos: output.stopped_on_eos,
        })
    }
}
