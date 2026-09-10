use crate::HuggingFaceLoadError;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeConfig {
    pub rope_type: String,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub layer_types: Vec<LayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub tie_word_embeddings: bool,
    pub rope_parameters: RopeConfig,
    pub eos_token_id: i32,
    pub mamba_ssm_dtype: String,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

impl Qwen35TextConfig {
    pub fn validate(&self) -> Result<(), HuggingFaceLoadError> {
        let fail = |message: &str| Err(HuggingFaceLoadError(message.into()));
        if self.model_type != "qwen3_5_text"
            || self.hidden_act != "silu"
            || self.rope_parameters.rope_type != "default"
            || self.attention_dropout != 0.0
            || self.mamba_ssm_dtype != "float32"
            || self.quantization_config.is_some()
        {
            return fail(
                "unsupported Qwen3.5 text architecture, activation, RoPE, dropout, state dtype or quantization",
            );
        }
        if [
            self.hidden_size,
            self.intermediate_size,
            self.vocab_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.max_position_embeddings,
            self.linear_conv_kernel_dim,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
        ]
        .contains(&0)
            || self.layer_types.len() != self.num_hidden_layers
        {
            return fail(
                "Qwen3.5 dimensions must be positive and layer_types must cover every layer",
            );
        }
        if self.num_attention_heads % self.num_key_value_heads != 0
            || self.linear_num_value_heads % self.linear_num_key_heads != 0
        {
            return fail("Qwen3.5 query/value head counts must be divisible by key head counts");
        }
        let rope = &self.rope_parameters;
        let rotary = self.head_dim as f64 * rope.partial_rotary_factor;
        if !rotary.is_finite()
            || rotary < 2.0
            || rotary > self.head_dim as f64
            || rotary.fract() != 0.0
            || rotary as usize % 2 != 0
            || !rope.rope_theta.is_finite()
            || rope.rope_theta <= 0.0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || self.eos_token_id < 0
            || self.eos_token_id as usize >= self.vocab_size
        {
            return fail("invalid Qwen3.5 rotary, normalization or EOS configuration");
        }
        for (a, b) in [
            (self.num_attention_heads, self.head_dim.saturating_mul(2)),
            (self.linear_num_key_heads, self.linear_key_head_dim),
            (self.linear_num_value_heads, self.linear_value_head_dim),
            (self.vocab_size, self.hidden_size),
        ] {
            if a.checked_mul(b).is_none_or(|n| n > u32::MAX as usize) {
                return fail("Qwen3.5 tensor dimensions exceed supported indexing");
            }
        }
        Ok(())
    }
}
