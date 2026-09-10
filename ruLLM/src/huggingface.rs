use crate::{
    GenerationError, GreedyGenerationConfig, LlamaConfig, LlamaForCausalLm, TokenGenerationOutput,
    generate_greedy,
};
use ruda_model::module::Param;
use ruda_tensor::api::backend::Backend;
use ruda_store::{ModuleSnapshot, PyTorchToRudaAdapter, SafetensorsStore};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

mod awq;
mod awq_model;
mod pipeline;
mod sampling;
pub(crate) mod checkpoint;

pub use awq::{AwqCheckpoint, AwqQuantizationConfig};
pub use awq_model::{LoadedHuggingFaceAwqQwen2, load_huggingface_awq_qwen2};
pub use pipeline::{
    HuggingFaceAwqQwen2Pipeline, load_huggingface_awq_qwen2_pipeline,
    load_huggingface_llama_pipeline, load_huggingface_qwen2_pipeline,
};

const CONFIG_FILE: &str = "config.json";
const SINGLE_WEIGHTS_FILE: &str = "model.safetensors";
const WEIGHTS_INDEX_FILE: &str = "model.safetensors.index.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
const GENERATION_CONFIG_FILE: &str = "generation_config.json";

#[derive(Debug, Clone, Deserialize)]
struct HuggingFaceLlamaConfig {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_theta: Option<f32>,
    rope_scaling: Option<serde_json::Value>,
    rope_parameters: Option<serde_json::Value>,
    hidden_act: Option<String>,
    attention_bias: Option<bool>,
    mlp_bias: Option<bool>,
    attention_dropout: Option<f64>,
    tie_word_embeddings: Option<bool>,
    head_dim: Option<usize>,
    eos_token_id: Option<HuggingFaceTokenIds>,
    use_sliding_window: Option<bool>,
    use_mrope: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum HuggingFaceTokenIds {
    One(i64),
    Many(Vec<i64>),
}

#[derive(Debug, Clone, Deserialize)]
struct HuggingFaceGenerationConfig {
    eos_token_id: Option<HuggingFaceTokenIds>,
}

#[derive(Debug, Clone, Deserialize)]
struct HuggingFaceTokenizerConfig {
    chat_template: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HuggingFaceLoadReport {
    pub config_path: PathBuf,
    pub weight_files: Vec<PathBuf>,
    pub applied_tensors: usize,
    pub tied_word_embeddings: bool,
}

#[derive(Debug)]
pub struct LoadedHuggingFaceLlama<B: Backend> {
    pub model: LlamaForCausalLm<B>,
    pub config: LlamaConfig,
    pub default_eos_token_ids: Vec<i32>,
    pub report: HuggingFaceLoadReport,
}

/// Strictly loaded Qwen2/Qwen2.5 causal language model. `config` contains the
/// common decoder geometry used by the paged Ruda runtime; the model itself was
/// initialized with Qwen-specific Q/K/V bias and rotary semantics.
#[derive(Debug)]
pub struct LoadedHuggingFaceQwen2<B: Backend> {
    pub model: LlamaForCausalLm<B>,
    pub config: LlamaConfig,
    pub default_eos_token_ids: Vec<i32>,
    pub report: HuggingFaceLoadReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedText {
    pub text: String,
    pub generated_text: String,
    pub token_ids: Vec<i32>,
    pub generated_token_ids: Vec<i32>,
    pub stopped_on_eos: bool,
}

/// Complete local Hugging Face Llama inference entry point: strict config and
/// Safetensors loading, the model's `tokenizer.json`, and cached text decoding.
#[derive(Debug)]
pub struct HuggingFaceLlamaPipeline<B: Backend> {
    pub loaded: LoadedHuggingFaceLlama<B>,
    pub tokenizer: Tokenizer,
}

#[derive(Debug)]
pub struct HuggingFaceQwen2Pipeline<B: Backend> {
    pub loaded: LoadedHuggingFaceQwen2<B>,
    pub tokenizer: Tokenizer,
    /// Exact Jinja template supplied by `tokenizer_config.json`. Ruda renders the
    /// plain-message/no-tools Qwen branch itself and rejects incompatible
    /// templates rather than silently applying a guessed prompt format.
    pub chat_template: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen2ChatRole {
    System,
    User,
    Assistant,
}

impl Qwen2ChatRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen2ChatMessage {
    pub role: Qwen2ChatRole,
    pub content: String,
}

impl Qwen2ChatMessage {
    pub fn new(role: Qwen2ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HuggingFaceLoadError(pub String);

impl Display for HuggingFaceLoadError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for HuggingFaceLoadError {}

/// Load an unquantized Hugging Face Llama directory into the Ruda model.
/// Both a single `model.safetensors` and the standard sharded
/// `model.safetensors.index.json` layout are supported. Every expected tensor
/// must be present and every source tensor must be consumed; incompatible
/// Llama variants are rejected instead of silently approximated.
pub fn load_huggingface_llama<B: Backend>(
    model_directory: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedHuggingFaceLlama<B>, HuggingFaceLoadError> {
    let model_directory = canonical_directory(model_directory.as_ref())?;
    let config_path = model_directory.join(CONFIG_FILE);
    let source: HuggingFaceLlamaConfig = read_json(&config_path)?;
    let (config, tied_word_embeddings, default_eos_token_ids) = convert_config(source)?;
    let weight_files = discover_weight_files(&model_directory)?;
    let mut model = LlamaForCausalLm::<B>::init(&config, device)
        .map_err(|error| HuggingFaceLoadError(error.to_string()))?;
    let expected = expected_tensor_paths(config.num_hidden_layers, tied_word_embeddings);
    let mut applied = BTreeSet::new();
    let mut unused = BTreeSet::new();

    for path in &weight_files {
        let mut store = SafetensorsStore::from_file(path)
            .with_key_remapping(r"^model\.", "")
            .with_key_remapping(
                r"^(.*\.(?:input_layernorm|post_attention_layernorm))\.weight$",
                "$1.gamma",
            )
            .with_key_remapping(r"^norm\.weight$", "norm.gamma")
            .with_from_adapter(PyTorchToRudaAdapter)
            .allow_partial(true)
            .validate(true);
        let result = model.load_from(&mut store).map_err(|error| {
            HuggingFaceLoadError(format!(
                "cannot load Hugging Face shard {}: {error}",
                path.display()
            ))
        })?;
        for name in result.applied {
            if !applied.insert(name.clone()) {
                return Err(HuggingFaceLoadError(format!(
                    "tensor {name} was supplied by more than one Safetensors shard"
                )));
            }
        }
        unused.extend(result.unused);
    }

    if tied_word_embeddings {
        let tied = model.embed_tokens.weight.val().transpose().detach();
        model.lm_head.weight = Param::from_tensor(tied);
    }
    let missing = expected.difference(&applied).cloned().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(HuggingFaceLoadError(format!(
            "Hugging Face model is missing Ruda Llama tensors: {}",
            missing.join(", ")
        )));
    }
    let unexpected = unused
        .into_iter()
        .filter(|path| !(tied_word_embeddings && path == "lm_head.weight"))
        .collect::<Vec<_>>();
    if !unexpected.is_empty() {
        return Err(HuggingFaceLoadError(format!(
            "Hugging Face model contains tensors not represented by Ruda Llama: {}",
            unexpected.join(", ")
        )));
    }
    B::sync(device).map_err(|error| {
        HuggingFaceLoadError(format!("Ruda weight upload did not complete: {error}"))
    })?;
    Ok(LoadedHuggingFaceLlama {
        model,
        config,
        default_eos_token_ids,
        report: HuggingFaceLoadReport {
            config_path,
            weight_files,
            applied_tensors: applied.len(),
            tied_word_embeddings,
        },
    })
}

/// Load an unquantized Hugging Face Qwen2 or Qwen2.5 directory. The loader
/// requires the architecture's three Q/K/V bias tensors, rejects sliding
/// attention and multimodal RoPE variants, and consumes every checkpoint
/// tensor instead of silently dropping unsupported state.
pub fn load_huggingface_qwen2<B: Backend>(
    model_directory: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedHuggingFaceQwen2<B>, HuggingFaceLoadError> {
    let model_directory = canonical_directory(model_directory.as_ref())?;
    let config_path = model_directory.join(CONFIG_FILE);
    let source: HuggingFaceLlamaConfig = read_json(&config_path)?;
    let (config, tied_word_embeddings, config_eos_token_ids) = convert_qwen2_config(source)?;
    let default_eos_token_ids =
        load_generation_eos_token_ids(&model_directory, config.vocab_size, config_eos_token_ids)?;
    let weight_files = discover_weight_files(&model_directory)?;
    let mut model = LlamaForCausalLm::<B>::init_qwen2(&config, device)
        .map_err(|error| HuggingFaceLoadError(error.to_string()))?;
    let expected = expected_qwen2_tensor_paths(config.num_hidden_layers, tied_word_embeddings);
    let mut applied = BTreeSet::new();
    let mut unused = BTreeSet::new();

    for path in &weight_files {
        let mut store = SafetensorsStore::from_file(path)
            .with_key_remapping(r"^model\.", "")
            .with_key_remapping(
                r"^(.*\.(?:input_layernorm|post_attention_layernorm))\.weight$",
                "$1.gamma",
            )
            .with_key_remapping(r"^norm\.weight$", "norm.gamma")
            .allow_partial(true)
            .validate(true);
        let result = model.load_from(&mut store).map_err(|error| {
            HuggingFaceLoadError(format!(
                "cannot load Hugging Face Qwen2 shard {}: {error}",
                path.display()
            ))
        })?;
        for name in result.applied {
            if !applied.insert(name.clone()) {
                return Err(HuggingFaceLoadError(format!(
                    "tensor {name} was supplied by more than one Safetensors shard"
                )));
            }
        }
        unused.extend(result.unused);
    }

    if tied_word_embeddings {
        let tied = model.embed_tokens.weight.val().transpose().detach();
        model.lm_head.weight = Param::from_tensor(tied);
    }
    let missing = expected.difference(&applied).cloned().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(HuggingFaceLoadError(format!(
            "Hugging Face model is missing Ruda Qwen2 tensors: {}",
            missing.join(", ")
        )));
    }
    let unexpected = unused
        .into_iter()
        .filter(|path| !(tied_word_embeddings && path == "lm_head.weight"))
        .collect::<Vec<_>>();
    if !unexpected.is_empty() {
        return Err(HuggingFaceLoadError(format!(
            "Hugging Face model contains tensors not represented by Ruda Qwen2: {}",
            unexpected.join(", ")
        )));
    }
    let mut rotary_caches: Vec<(ruda_tensor::api::DType, ruda_nn::RotaryEncoding<B>)> = Vec::new();
    for layer in &mut model.layers {
        let dtype = layer.self_attn.q_proj.weight.val().dtype();
        if layer.self_attn.rope.freq_complex.dtype() != dtype {
            let rope = if let Some((_, rope)) = rotary_caches.iter().find(|(kind, _)| *kind == dtype) {
                rope.clone()
            } else {
                let rope = crate::llama::rope::qwen2_rope_with_dtype::<B>(&config, device, dtype);
                rotary_caches.push((dtype, rope.clone()));
                rope
            };
            layer.self_attn.rope = rope;
        }
    }
    B::sync(device).map_err(|error| {
        HuggingFaceLoadError(format!("Ruda Qwen2 weight upload did not complete: {error}"))
    })?;
    Ok(LoadedHuggingFaceQwen2 {
        model,
        config,
        default_eos_token_ids,
        report: HuggingFaceLoadReport {
            config_path,
            weight_files,
            applied_tensors: applied.len(),
            tied_word_embeddings,
        },
    })
}

impl<B: Backend> LoadedHuggingFaceLlama<B> {
    /// Decode token IDs with the loaded model. An empty EOS list inherits the
    /// exact `eos_token_id` value from Hugging Face `config.json`.
    pub fn generate_tokens(
        &self,
        prompt_token_ids: &[i32],
        mut generation: GreedyGenerationConfig,
        device: &B::Device,
    ) -> Result<TokenGenerationOutput, GenerationError> {
        if generation.eos_token_ids.is_empty() {
            generation.eos_token_ids = self.default_eos_token_ids.clone();
        }
        generate_greedy(
            &self.model,
            &self.config,
            prompt_token_ids,
            &generation,
            device,
        )
    }
}

impl<B: Backend> LoadedHuggingFaceQwen2<B> {
    /// Decode with Qwen's architecture-specific model. An empty EOS list
    /// inherits the exact scalar or list value from `config.json`.
    pub fn generate_tokens(
        &self,
        prompt_token_ids: &[i32],
        mut generation: GreedyGenerationConfig,
        device: &B::Device,
    ) -> Result<TokenGenerationOutput, GenerationError> {
        if generation.eos_token_ids.is_empty() {
            generation.eos_token_ids = self.default_eos_token_ids.clone();
        }
        generate_greedy(
            &self.model,
            &self.config,
            prompt_token_ids,
            &generation,
            device,
        )
    }
}

fn convert_config(
    source: HuggingFaceLlamaConfig,
) -> Result<(LlamaConfig, bool, Vec<i32>), HuggingFaceLoadError> {
    if source
        .model_type
        .as_deref()
        .is_some_and(|value| value != "llama")
    {
        return Err(HuggingFaceLoadError(format!(
            "model_type {:?} is not Llama",
            source.model_type
        )));
    }
    if source.architectures.as_ref().is_some_and(|architectures| {
        !architectures
            .iter()
            .any(|architecture| architecture == "LlamaForCausalLM")
    }) {
        return Err(HuggingFaceLoadError(format!(
            "architectures {:?} do not contain LlamaForCausalLM",
            source.architectures
        )));
    }
    if source
        .hidden_act
        .as_deref()
        .is_some_and(|value| value != "silu")
    {
        return Err(HuggingFaceLoadError(format!(
            "Ruda Llama requires hidden_act=silu, got {:?}",
            source.hidden_act
        )));
    }
    if source.attention_bias.unwrap_or(false) || source.mlp_bias.unwrap_or(false) {
        return Err(HuggingFaceLoadError(
            "Ruda Llama does not implement attention or MLP projection biases".into(),
        ));
    }
    if source.attention_dropout.unwrap_or(0.0) != 0.0 {
        return Err(HuggingFaceLoadError(
            "Ruda Llama currently requires attention_dropout=0".into(),
        ));
    }
    if source
        .rope_scaling
        .as_ref()
        .is_some_and(|value| !value.is_null())
    {
        return Err(HuggingFaceLoadError(
            "Ruda Llama does not yet implement Hugging Face rope_scaling variants".into(),
        ));
    }
    let num_kv_heads = source
        .num_key_value_heads
        .unwrap_or(source.num_attention_heads);
    let config = LlamaConfig {
        vocab_size: source.vocab_size,
        d_model: source.hidden_size,
        d_ff: source.intermediate_size,
        num_hidden_layers: source.num_hidden_layers,
        num_query_heads: source.num_attention_heads,
        num_kv_heads,
        max_sequence_length: source.max_position_embeddings,
        rms_norm_epsilon: source.rms_norm_eps,
        rope_theta: source.rope_theta.unwrap_or(10_000.0),
    };
    config
        .validate()
        .map_err(|error| HuggingFaceLoadError(error.to_string()))?;
    if let Some(head_dim) = source.head_dim
        && head_dim != config.head_dimension()
    {
        return Err(HuggingFaceLoadError(format!(
            "head_dim {head_dim} does not equal hidden_size/num_attention_heads ({})",
            config.head_dimension()
        )));
    }
    let eos_token_ids = convert_token_ids(source.eos_token_id, config.vocab_size)?;
    Ok((
        config,
        source.tie_word_embeddings.unwrap_or(false),
        eos_token_ids,
    ))
}

fn convert_qwen2_config(
    source: HuggingFaceLlamaConfig,
) -> Result<(LlamaConfig, bool, Vec<i32>), HuggingFaceLoadError> {
    if source.model_type.as_deref() != Some("qwen2") {
        return Err(HuggingFaceLoadError(format!(
            "model_type {:?} is not Qwen2",
            source.model_type
        )));
    }
    if source.architectures.as_ref().is_none_or(|architectures| {
        !architectures
            .iter()
            .any(|architecture| architecture == "Qwen2ForCausalLM")
    }) {
        return Err(HuggingFaceLoadError(format!(
            "architectures {:?} do not contain Qwen2ForCausalLM",
            source.architectures
        )));
    }
    if source.hidden_act.as_deref().unwrap_or("silu") != "silu" {
        return Err(HuggingFaceLoadError(format!(
            "Ruda Qwen2 requires hidden_act=silu, got {:?}",
            source.hidden_act
        )));
    }
    if source.mlp_bias.unwrap_or(false) {
        return Err(HuggingFaceLoadError(
            "Ruda Qwen2 does not implement MLP projection biases".into(),
        ));
    }
    // Qwen2Config defaults `attention_bias` to true, and official Qwen2.5
    // checkpoints commonly omit the field while still containing all three
    // bias tensors. Only an explicit false is incompatible.
    if source.attention_bias == Some(false) {
        return Err(HuggingFaceLoadError(format!(
            "Ruda Qwen2 requires attention_bias=true for Q/K/V projections, got {:?}",
            source.attention_bias
        )));
    }
    if source.attention_dropout.unwrap_or(0.0) != 0.0 {
        return Err(HuggingFaceLoadError(
            "Ruda Qwen2 currently requires attention_dropout=0".into(),
        ));
    }
    if source.use_sliding_window.unwrap_or(false) {
        return Err(HuggingFaceLoadError(
            "Ruda Qwen2 does not yet implement sliding-window attention".into(),
        ));
    }
    if source.use_mrope.unwrap_or(false) {
        return Err(HuggingFaceLoadError(
            "Ruda Qwen2 does not implement multimodal RoPE".into(),
        ));
    }
    if source
        .rope_scaling
        .as_ref()
        .is_some_and(|value| !value.is_null())
        || source
            .rope_parameters
            .as_ref()
            .is_some_and(|value| !value.is_null())
    {
        return Err(HuggingFaceLoadError(
            "Ruda Qwen2 currently requires default, unscaled RoPE".into(),
        ));
    }
    let num_kv_heads = source
        .num_key_value_heads
        .unwrap_or(source.num_attention_heads);
    let config = LlamaConfig {
        vocab_size: source.vocab_size,
        d_model: source.hidden_size,
        d_ff: source.intermediate_size,
        num_hidden_layers: source.num_hidden_layers,
        num_query_heads: source.num_attention_heads,
        num_kv_heads,
        max_sequence_length: source.max_position_embeddings,
        rms_norm_epsilon: source.rms_norm_eps,
        rope_theta: source.rope_theta.unwrap_or(10_000.0),
    };
    config
        .validate()
        .map_err(|error| HuggingFaceLoadError(error.to_string()))?;
    if let Some(head_dim) = source.head_dim
        && head_dim != config.head_dimension()
    {
        return Err(HuggingFaceLoadError(format!(
            "head_dim {head_dim} does not equal hidden_size/num_attention_heads ({})",
            config.head_dimension()
        )));
    }
    let eos_token_ids = convert_token_ids(source.eos_token_id, config.vocab_size)?;
    Ok((
        config,
        source.tie_word_embeddings.unwrap_or(false),
        eos_token_ids,
    ))
}

fn load_generation_eos_token_ids(
    model_directory: &Path,
    vocabulary: usize,
    config_eos_token_ids: Vec<i32>,
) -> Result<Vec<i32>, HuggingFaceLoadError> {
    let path = model_directory.join(GENERATION_CONFIG_FILE);
    if !path.is_file() {
        return Ok(config_eos_token_ids);
    }
    let generation: HuggingFaceGenerationConfig = read_json(&path)?;
    match generation.eos_token_id {
        Some(eos_token_id) => convert_token_ids(Some(eos_token_id), vocabulary),
        None => Ok(config_eos_token_ids),
    }
}

fn validate_qwen2_chat_template(template: &str) -> Result<(), HuggingFaceLoadError> {
    for required in [
        "<|im_start|>",
        "<|im_end|>",
        "message.role",
        "message.content",
        "add_generation_prompt",
    ] {
        if !template.contains(required) {
            return Err(HuggingFaceLoadError(format!(
                "Qwen chat_template is not the supported plain-message ChatML form: missing {required:?}"
            )));
        }
    }
    Ok(())
}

fn append_qwen2_chat_message(prompt: &mut String, message: &Qwen2ChatMessage) {
    prompt.push_str("<|im_start|>");
    prompt.push_str(message.role.as_str());
    prompt.push('\n');
    prompt.push_str(&message.content);
    prompt.push_str("<|im_end|>\n");
}

fn render_qwen2_chat_messages(
    messages: &[Qwen2ChatMessage],
    add_generation_prompt: bool,
) -> Result<String, HuggingFaceLoadError> {
    if messages.is_empty() {
        return Err(HuggingFaceLoadError(
            "Qwen chat requires at least one message".into(),
        ));
    }
    let mut prompt = String::new();
    let mut first_message = 0;
    if messages[0].role == Qwen2ChatRole::System {
        append_qwen2_chat_message(&mut prompt, &messages[0]);
        first_message = 1;
    } else {
        append_qwen2_chat_message(
            &mut prompt,
            &Qwen2ChatMessage::new(Qwen2ChatRole::System, "You are a helpful assistant."),
        );
    }
    for message in &messages[first_message..] {
        append_qwen2_chat_message(&mut prompt, message);
    }
    if add_generation_prompt {
        prompt.push_str("<|im_start|>assistant\n");
    }
    Ok(prompt)
}

fn convert_token_ids(
    source: Option<HuggingFaceTokenIds>,
    vocabulary: usize,
) -> Result<Vec<i32>, HuggingFaceLoadError> {
    let values = match source {
        None => return Ok(Vec::new()),
        Some(HuggingFaceTokenIds::One(value)) => vec![value],
        Some(HuggingFaceTokenIds::Many(values)) => values,
    };
    let mut result = BTreeSet::new();
    for value in values {
        let token = i32::try_from(value).map_err(|_| {
            HuggingFaceLoadError(format!("eos_token_id {value} does not fit an i32 token id"))
        })?;
        if token < 0 || token as usize >= vocabulary {
            return Err(HuggingFaceLoadError(format!(
                "eos_token_id {token} is outside vocabulary [0, {vocabulary})"
            )));
        }
        result.insert(token);
    }
    Ok(result.into_iter().collect())
}

fn discover_weight_files(directory: &Path) -> Result<Vec<PathBuf>, HuggingFaceLoadError> {
    let single = directory.join(SINGLE_WEIGHTS_FILE);
    if single.is_file() {
        return Ok(vec![single]);
    }
    let index_path = directory.join(WEIGHTS_INDEX_FILE);
    let index: SafetensorsIndex = read_json(&index_path)?;
    if index.weight_map.is_empty() {
        return Err(HuggingFaceLoadError(format!(
            "{} contains an empty weight_map",
            index_path.display()
        )));
    }
    let names = index.weight_map.into_values().collect::<BTreeSet<_>>();
    let mut files = Vec::with_capacity(names.len());
    for name in &names {
        let relative = Path::new(name);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(HuggingFaceLoadError(format!(
                "Safetensors index contains an unsafe shard path: {name}"
            )));
        }
        let path = directory.join(relative);
        if !path.is_file() {
            return Err(HuggingFaceLoadError(format!(
                "Safetensors shard does not exist: {}",
                path.display()
            )));
        }
        let canonical = path.canonicalize().map_err(io_error)?;
        if !canonical.starts_with(directory) {
            return Err(HuggingFaceLoadError(format!(
                "Safetensors shard resolves outside the model directory: {}",
                canonical.display()
            )));
        }
        files.push(canonical);
    }
    Ok(files)
}

fn expected_tensor_paths(layers: usize, tied_word_embeddings: bool) -> BTreeSet<String> {
    let mut paths = BTreeSet::from(["embed_tokens.weight".to_string(), "norm.gamma".to_string()]);
    if !tied_word_embeddings {
        paths.insert("lm_head.weight".to_string());
    }
    for layer in 0..layers {
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.gamma",
            "post_attention_layernorm.gamma",
        ] {
            paths.insert(format!("layers.{layer}.{suffix}"));
        }
    }
    paths
}

fn expected_qwen2_tensor_paths(layers: usize, tied_word_embeddings: bool) -> BTreeSet<String> {
    let mut paths = expected_tensor_paths(layers, tied_word_embeddings);
    for layer in 0..layers {
        for projection in ["q_proj", "k_proj", "v_proj"] {
            paths.insert(format!("layers.{layer}.self_attn.{projection}.bias"));
        }
    }
    paths
}

fn canonical_directory(path: &Path) -> Result<PathBuf, HuggingFaceLoadError> {
    let path = path.canonicalize().map_err(io_error)?;
    if !path.is_dir() {
        return Err(HuggingFaceLoadError(format!(
            "Hugging Face model path is not a directory: {}",
            path.display()
        )));
    }
    Ok(path)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, HuggingFaceLoadError> {
    let bytes = fs::read(path).map_err(|error| {
        HuggingFaceLoadError(format!("cannot read {}: {error}", path.display()))
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|error| HuggingFaceLoadError(format!("cannot parse {}: {error}", path.display())))
}

fn io_error(error: std::io::Error) -> HuggingFaceLoadError {
    HuggingFaceLoadError(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN_05B_CONFIG: &str = r#"{
        "architectures": ["Qwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "eos_token_id": 151643,
        "hidden_act": "silu",
        "hidden_size": 896,
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
        "model_type": "qwen2",
        "num_attention_heads": 14,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": true,
        "use_mrope": false,
        "use_sliding_window": false,
        "vocab_size": 151936
    }"#;

    #[test]
    fn qwen_05b_config_maps_exact_geometry_and_bias_paths() {
        let source: HuggingFaceLlamaConfig = serde_json::from_str(QWEN_05B_CONFIG).unwrap();
        let (config, tied, eos) = convert_qwen2_config(source).unwrap();
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.d_model, 896);
        assert_eq!(config.d_ff, 4864);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_query_heads, 14);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dimension(), 64);
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert!(tied);
        assert_eq!(eos, vec![151643]);
        let expected = expected_qwen2_tensor_paths(24, tied);
        assert!(expected.contains("layers.0.self_attn.q_proj.bias"));
        assert!(expected.contains("layers.23.self_attn.k_proj.bias"));
        assert!(expected.contains("layers.23.self_attn.v_proj.bias"));
        assert!(!expected.contains("lm_head.weight"));
    }

    #[test]
    fn qwen_adapter_rejects_unimplemented_sliding_attention() {
        let mut value: serde_json::Value = serde_json::from_str(QWEN_05B_CONFIG).unwrap();
        value["use_sliding_window"] = serde_json::Value::Bool(true);
        let source = serde_json::from_value(value).unwrap();
        let error = convert_qwen2_config(source).unwrap_err();
        assert!(error.0.contains("sliding-window"));
    }

    #[test]
    fn qwen_chat_renders_official_no_tools_branch() {
        let prompt =
            render_qwen2_chat_messages(&[Qwen2ChatMessage::new(Qwen2ChatRole::User, "你好")], true)
                .unwrap();
        assert_eq!(
            prompt,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen_adapter_requires_qkv_bias_declaration() {
        let mut value: serde_json::Value = serde_json::from_str(QWEN_05B_CONFIG).unwrap();
        value["attention_bias"] = serde_json::Value::Bool(false);
        let source = serde_json::from_value(value).unwrap();
        let error = convert_qwen2_config(source).unwrap_err();
        assert!(error.0.contains("attention_bias=true"));
    }
}

#[cfg(all(test, feature = "nvidia"))]
#[path = "huggingface/checkpoint_tests.rs"]
mod checkpoint_tests;
