use super::*;
use crate::SamplingGenerationConfig;
use ruda_tensor::DeviceOps;
use ruda_tensor_device::DeviceRuntime;

#[derive(Debug)]
pub struct HuggingFaceAwqQwen2Pipeline<R: DeviceRuntime>
where
    R::Device: DeviceOps,
{
    pub loaded: LoadedHuggingFaceAwqQwen2<R>,
    pub tokenizer: Tokenizer,
    pub chat_template: String,
}

/// Load Qwen2/Qwen2.5 weights and its exact local `tokenizer.json`.
pub fn load_huggingface_awq_qwen2_pipeline<R: DeviceRuntime>(
    model_directory: impl AsRef<Path>,
    device: &R::Device,
) -> Result<HuggingFaceAwqQwen2Pipeline<R>, HuggingFaceLoadError>
where
    R::Device: DeviceOps,
{
    let directory = canonical_directory(model_directory.as_ref())?;
    let tokenizer = load_tokenizer(&directory)?;
    let chat_template = load_qwen_chat_template(&directory)?;
    let loaded = load_huggingface_awq_qwen2::<R>(&directory, device)?;
    validate_qwen_vocabulary(&tokenizer, loaded.config.vocab_size)?;
    Ok(HuggingFaceAwqQwen2Pipeline {
        loaded,
        tokenizer,
        chat_template,
    })
}

impl<R: DeviceRuntime> HuggingFaceAwqQwen2Pipeline<R>
where
    R::Device: DeviceOps,
{
    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<i32>, HuggingFaceLoadError> {
        encode(&self.tokenizer, text, add_special_tokens)
    }

    pub fn decode(
        &self,
        token_ids: &[i32],
        skip_special_tokens: bool,
    ) -> Result<String, HuggingFaceLoadError> {
        decode(&self.tokenizer, token_ids, skip_special_tokens)
    }

    /// Render the no-tools, plain-text branch of Qwen2/Qwen2.5's ChatML
    /// template. Tool calls are deliberately not approximated by this API.
    pub fn render_chat(
        &self,
        messages: &[Qwen2ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, HuggingFaceLoadError> {
        render_qwen2_chat_messages(messages, add_generation_prompt)
    }

    pub fn encode_chat(
        &self,
        messages: &[Qwen2ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<Vec<i32>, HuggingFaceLoadError> {
        let prompt = self.render_chat(messages, add_generation_prompt)?;
        self.encode(&prompt, false)
    }

    pub fn generate_text(
        &self,
        prompt: &str,
        generation: GreedyGenerationConfig,
        add_special_tokens: bool,
        skip_special_tokens: bool,
        device: &R::Device,
    ) -> Result<GeneratedText, HuggingFaceLoadError> {
        let prompt_ids = self.encode(prompt, add_special_tokens)?;
        let output = self
            .loaded
            .generate_tokens(&prompt_ids, generation, device)
            .map_err(|error| HuggingFaceLoadError(error.to_string()))?;
        let text = self.decode(&output.token_ids, skip_special_tokens)?;
        let generated_text = self.decode(&output.generated_token_ids, skip_special_tokens)?;
        Ok(GeneratedText {
            text,
            generated_text,
            token_ids: output.token_ids,
            generated_token_ids: output.generated_token_ids,
            stopped_on_eos: output.stopped_on_eos,
        })
    }
}

impl<R: DeviceRuntime> HuggingFaceAwqQwen2Pipeline<R>
where
    R::Device: DeviceOps,
{
    /// Tokenize, run cached sampled decoding, and decode the generated text.
    pub fn generate_text_sampled(
        &self,
        prompt: &str,
        generation: SamplingGenerationConfig,
        add_special_tokens: bool,
        skip_special_tokens: bool,
        device: &R::Device,
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
