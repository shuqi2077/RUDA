mod awq;
pub use awq::{HuggingFaceAwqQwen2Pipeline, load_huggingface_awq_qwen2_pipeline};

use super::*;

/// Load the model and the local Hugging Face `tokenizer.json` without making
/// network requests or substituting another tokenizer implementation.
pub fn load_huggingface_llama_pipeline<B: Backend>(
    model_directory: impl AsRef<Path>,
    device: &B::Device,
) -> Result<HuggingFaceLlamaPipeline<B>, HuggingFaceLoadError> {
    let directory = canonical_directory(model_directory.as_ref())?;
    let tokenizer = load_tokenizer(&directory)?;
    let loaded = load_huggingface_llama::<B>(&directory, device)?;
    if tokenizer.get_vocab_size(true) != loaded.config.vocab_size {
        return Err(HuggingFaceLoadError(format!(
            "tokenizer vocabulary {} does not equal model vocabulary {}",
            tokenizer.get_vocab_size(true),
            loaded.config.vocab_size
        )));
    }
    Ok(HuggingFaceLlamaPipeline { loaded, tokenizer })
}

/// Load Qwen2/Qwen2.5 weights and its exact local `tokenizer.json`.
pub fn load_huggingface_qwen2_pipeline<B: Backend>(
    model_directory: impl AsRef<Path>,
    device: &B::Device,
) -> Result<HuggingFaceQwen2Pipeline<B>, HuggingFaceLoadError> {
    let directory = canonical_directory(model_directory.as_ref())?;
    let tokenizer = load_tokenizer(&directory)?;
    let chat_template = load_qwen_chat_template(&directory)?;
    let loaded = load_huggingface_qwen2::<B>(&directory, device)?;
    validate_qwen_vocabulary(&tokenizer, loaded.config.vocab_size)?;
    Ok(HuggingFaceQwen2Pipeline {
        loaded,
        tokenizer,
        chat_template,
    })
}

impl<B: Backend> HuggingFaceLlamaPipeline<B> {
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

    /// Tokenize a batch-one prompt, run cached greedy generation, and decode
    /// both the full sequence and newly generated suffix.
    pub fn generate_text(
        &self,
        prompt: &str,
        generation: GreedyGenerationConfig,
        add_special_tokens: bool,
        skip_special_tokens: bool,
        device: &B::Device,
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

impl<B: Backend> HuggingFaceQwen2Pipeline<B> {
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
        device: &B::Device,
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

fn load_tokenizer(directory: &Path) -> Result<Tokenizer, HuggingFaceLoadError> {
    let tokenizer_path = directory.join(TOKENIZER_FILE);
    if !tokenizer_path.is_file() {
        return Err(HuggingFaceLoadError(format!(
            "Hugging Face tokenizer does not exist: {}",
            tokenizer_path.display()
        )));
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
        HuggingFaceLoadError(format!(
            "cannot load Hugging Face tokenizer {}: {error}",
            tokenizer_path.display()
        ))
    })?;
    Ok(tokenizer)
}

fn load_qwen_chat_template(directory: &Path) -> Result<String, HuggingFaceLoadError> {
    let tokenizer_config_path = directory.join(TOKENIZER_CONFIG_FILE);
    let tokenizer_config: HuggingFaceTokenizerConfig = read_json(&tokenizer_config_path)?;
    let chat_template = tokenizer_config.chat_template.ok_or_else(|| {
        HuggingFaceLoadError(format!(
            "Qwen tokenizer config {} does not contain chat_template",
            tokenizer_config_path.display()
        ))
    })?;
    validate_qwen2_chat_template(&chat_template)?;
    Ok(chat_template)
}

fn validate_qwen_vocabulary(
    tokenizer: &Tokenizer,
    vocab_size: usize,
) -> Result<(), HuggingFaceLoadError> {
    let vocabulary = tokenizer.get_vocab(true);
    if let Some((token, id)) = vocabulary
        .iter()
        .find(|(_, id)| **id as usize >= vocab_size)
    {
        return Err(HuggingFaceLoadError(format!(
            "tokenizer token {token:?} has id {id}, outside model vocabulary [0, {})",
            vocab_size
        )));
    }
    Ok(())
}

fn encode(
    tokenizer: &Tokenizer,
    text: &str,
    add_special_tokens: bool,
) -> Result<Vec<i32>, HuggingFaceLoadError> {
    let encoding = tokenizer
        .encode(text, add_special_tokens)
        .map_err(|error| HuggingFaceLoadError(format!("tokenizer encode failed: {error}")))?;
    encoding
        .get_ids()
        .iter()
        .map(|&token| {
            i32::try_from(token).map_err(|_| {
                HuggingFaceLoadError(format!("tokenizer produced token {token} above i32::MAX"))
            })
        })
        .collect()
}

fn decode(
    tokenizer: &Tokenizer,
    token_ids: &[i32],
    skip_special_tokens: bool,
) -> Result<String, HuggingFaceLoadError> {
    let ids = token_ids
        .iter()
        .map(|&token| {
            u32::try_from(token).map_err(|_| {
                HuggingFaceLoadError(format!("cannot decode negative token id {token}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    tokenizer
        .decode(&ids, skip_special_tokens)
        .map_err(|error| HuggingFaceLoadError(format!("tokenizer decode failed: {error}")))
}
