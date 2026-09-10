use ruda_model::module::{Module, Param};
use ruda_nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout, RmsNorm, RmsNormConfig,
    RotaryEncoding, RotaryEncodingConfig,
};
use ruda_tensor::api::activation::{silu, softmax};
use ruda_tensor::api::backend::Backend;
use ruda_tensor::api::module::{attention, linear};
use ruda_tensor::api::{DType, Int, Tensor, TensorData};
use ruda_tensor::{DeviceOps, ops::AttentionModuleOptions};
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};
use ruda::runtime::server::ComputeServer;
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::ruda_inference;

pub(crate) mod awq;
pub(crate) mod rope;
use rope::qwen2_rope;

const PACKED_TRAINING_MIN_TOKENS: usize = 32;
const FLASH_ATTENTION_MIN_SEQUENCE: usize = 512;

/// Architecture parameters for a decoder-only Llama model.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub num_hidden_layers: usize,
    pub num_query_heads: usize,
    pub num_kv_heads: usize,
    pub max_sequence_length: usize,
    pub rms_norm_epsilon: f64,
    pub rope_theta: f32,
}

impl LlamaConfig {
    pub fn validate(&self) -> Result<(), LlamaConfigError> {
        if self.vocab_size == 0
            || self.d_model == 0
            || self.d_ff == 0
            || self.num_hidden_layers == 0
            || self.num_query_heads == 0
            || self.num_kv_heads == 0
            || self.max_sequence_length == 0
        {
            return Err(LlamaConfigError::InvalidDimension);
        }
        if !self.d_model.is_multiple_of(self.num_query_heads) {
            return Err(LlamaConfigError::HeadDimension);
        }
        let head_dimension = self.d_model / self.num_query_heads;
        if !head_dimension.is_multiple_of(2) {
            return Err(LlamaConfigError::RotaryDimension);
        }
        if !self.num_query_heads.is_multiple_of(self.num_kv_heads) {
            return Err(LlamaConfigError::GroupedQueryHeads);
        }
        if !self.rms_norm_epsilon.is_finite() || self.rms_norm_epsilon <= 0.0 {
            return Err(LlamaConfigError::RmsNormEpsilon);
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(LlamaConfigError::RopeTheta);
        }
        Ok(())
    }

    pub fn head_dimension(&self) -> usize {
        self.d_model / self.num_query_heads
    }

}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlamaConfigError {
    InvalidDimension,
    HeadDimension,
    RotaryDimension,
    GroupedQueryHeads,
    RmsNormEpsilon,
    RopeTheta,
}

impl Display for LlamaConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidDimension => "all Llama dimensions must be non-zero",
            Self::HeadDimension => "d_model must be divisible by num_query_heads",
            Self::RotaryDimension => "the query head dimension must be even for RoPE",
            Self::GroupedQueryHeads => "num_query_heads must be divisible by num_kv_heads",
            Self::RmsNormEpsilon => "RMS norm epsilon must be finite and positive",
            Self::RopeTheta => "RoPE theta must be finite and positive",
        };
        formatter.write_str(message)
    }
}

impl Error for LlamaConfigError {}

/// One layer of K/V state. Keys are stored after RoPE in `[B, H_kv, S, D_h]` order.
#[derive(Clone, Debug)]
pub struct LlamaLayerCache<B: Backend> {
    key: Option<Tensor<B, 4>>,
    value: Option<Tensor<B, 4>>,
    sequence_length: usize,
    storage_capacity: usize,
    max_sequence_length: usize,
}

impl<B: Backend> LlamaLayerCache<B> {
    fn new(max_sequence_length: usize) -> Self {
        Self {
            key: None,
            value: None,
            sequence_length: 0,
            storage_capacity: 0,
            max_sequence_length,
        }
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    fn append(&mut self, key: Tensor<B, 4>, value: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let appended = key.dims()[2];
        let key = match self.key.take() {
            Some(previous) => Tensor::cat(vec![previous, key], 2),
            None => key,
        };
        let value = match self.value.take() {
            Some(previous) => Tensor::cat(vec![previous, value], 2),
            None => value,
        };
        self.sequence_length += appended;
        self.storage_capacity = self.sequence_length;
        self.key = Some(key.clone());
        self.value = Some(value.clone());
        (key, value)
    }

    /// Append into a geometrically growing physical buffer. The logical prefix
    /// remains contiguous, while capacity growth only copies history at chunk
    /// boundaries instead of once for every decoded token.
    fn append_growable(
        &mut self,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, usize) {
        let [batch, kv_heads, appended, head_dimension] = key.dims();
        assert_eq!(value.dims(), key.dims());
        assert_eq!(value.dtype(), key.dtype());
        let device = key.device();
        let dtype = key.dtype();
        let (mut key_storage, mut value_storage, start, end) =
            self.reserve_growable(batch, kv_heads, appended, head_dimension, &device, dtype);
        key_storage.inplace(|storage| {
            storage.slice_assign([0..batch, 0..kv_heads, start..end, 0..head_dimension], key)
        });
        value_storage.inplace(|storage| {
            storage.slice_assign(
                [0..batch, 0..kv_heads, start..end, 0..head_dimension],
                value,
            )
        });
        self.commit_growable(key_storage, value_storage, end)
    }

    fn reserve_growable(
        &mut self,
        batch: usize,
        kv_heads: usize,
        appended: usize,
        head_dimension: usize,
        device: &B::Device,
        dtype: DType,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, usize, usize) {
        const INITIAL_CAPACITY: usize = 128;

        let end = self
            .sequence_length
            .checked_add(appended)
            .expect("KV cache position overflow");
        assert!(
            end <= self.max_sequence_length,
            "KV cache capacity exceeded"
        );

        if self.storage_capacity < end {
            let mut capacity = self.storage_capacity.max(INITIAL_CAPACITY);
            while capacity < end {
                capacity = capacity.checked_mul(2).expect("KV cache capacity overflow");
            }
            capacity = capacity.min(self.max_sequence_length);

            let mut key_storage =
                Tensor::empty([batch, kv_heads, capacity, head_dimension], (device, dtype));
            let mut value_storage =
                Tensor::empty([batch, kv_heads, capacity, head_dimension], (device, dtype));

            if let (Some(previous_key), Some(previous_value)) = (self.key.take(), self.value.take())
            {
                let previous_length = self.sequence_length;
                let previous_key = previous_key.slice([
                    0..batch,
                    0..kv_heads,
                    0..previous_length,
                    0..head_dimension,
                ]);
                let previous_value = previous_value.slice([
                    0..batch,
                    0..kv_heads,
                    0..previous_length,
                    0..head_dimension,
                ]);
                key_storage.inplace(|storage| {
                    storage.slice_assign(
                        [0..batch, 0..kv_heads, 0..previous_length, 0..head_dimension],
                        previous_key,
                    )
                });
                value_storage.inplace(|storage| {
                    storage.slice_assign(
                        [0..batch, 0..kv_heads, 0..previous_length, 0..head_dimension],
                        previous_value,
                    )
                });
            }
            self.key = Some(key_storage);
            self.value = Some(value_storage);
            self.storage_capacity = capacity;
        }

        let start = self.sequence_length;
        let key_storage = self.key.take().expect("KV key storage must exist");
        let value_storage = self.value.take().expect("KV value storage must exist");
        assert_eq!(
            key_storage.dims(),
            [batch, kv_heads, self.storage_capacity, head_dimension]
        );
        assert_eq!(value_storage.dims(), key_storage.dims());
        (key_storage, value_storage, start, end)
    }

    fn commit_growable(
        &mut self,
        key_storage: Tensor<B, 4>,
        value_storage: Tensor<B, 4>,
        end: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, usize) {
        assert!(end >= self.sequence_length && end <= self.storage_capacity);
        self.sequence_length = end;
        self.key = Some(key_storage.clone());
        self.value = Some(value_storage.clone());
        (key_storage, value_storage, end)
    }
}

/// Per-layer KV cache and the absolute RoPE position for autoregressive decoding.
#[derive(Clone, Debug)]
pub struct LlamaKvCache<B: Backend> {
    layers: Vec<LlamaLayerCache<B>>,
    position: usize,
    max_sequence_length: usize,
}

impl<B: Backend> LlamaKvCache<B> {
    pub fn new(num_hidden_layers: usize, max_sequence_length: usize) -> Self {
        assert!(
            num_hidden_layers > 0,
            "cache must contain at least one layer"
        );
        assert!(max_sequence_length > 0, "cache capacity must be non-zero");
        Self {
            layers: (0..num_hidden_layers)
                .map(|_| LlamaLayerCache::new(max_sequence_length))
                .collect(),
            position: 0,
            max_sequence_length,
        }
    }

    pub const fn position(&self) -> usize {
        self.position
    }

    pub const fn capacity(&self) -> usize {
        self.max_sequence_length
    }
}

#[derive(Module, Debug)]
pub struct LlamaAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub rope: RotaryEncoding<B>,
    #[module(skip)]
    rotary_layout: RotaryLayout,
    num_query_heads: usize,
    num_kv_heads: usize,
    head_dimension: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RotaryLayout {
    Interleaved,
    HalfSplit,
}

impl<B: Backend> LlamaAttention<B> {
    fn init(config: &LlamaConfig, device: &B::Device) -> Self {
        let rope = RotaryEncodingConfig::new(config.max_sequence_length, config.head_dimension())
            .with_theta(config.rope_theta)
            .init(device);
        Self::init_with_options(
            config,
            false,
            RotaryLayout::Interleaved,
            LinearLayout::Row,
            rope,
            device,
        )
    }

    fn init_qwen2_with_rope(
        config: &LlamaConfig,
        rope: RotaryEncoding<B>,
        device: &B::Device,
    ) -> Self {
        Self::init_with_options(
            config,
            true,
            RotaryLayout::HalfSplit,
            LinearLayout::Col,
            rope,
            device,
        )
    }

    fn init_with_options(
        config: &LlamaConfig,
        qkv_bias: bool,
        rotary_layout: RotaryLayout,
        linear_layout: LinearLayout,
        rope: RotaryEncoding<B>,
        device: &B::Device,
    ) -> Self {
        let head_dimension = config.head_dimension();
        let kv_dimension = config.num_kv_heads * head_dimension;
        Self {
            q_proj: LinearConfig::new(config.d_model, config.d_model)
                .with_bias(qkv_bias)
                .with_layout(linear_layout)
                .init(device),
            k_proj: LinearConfig::new(config.d_model, kv_dimension)
                .with_bias(qkv_bias)
                .with_layout(linear_layout)
                .init(device),
            v_proj: LinearConfig::new(config.d_model, kv_dimension)
                .with_bias(qkv_bias)
                .with_layout(linear_layout)
                .init(device),
            o_proj: LinearConfig::new(config.d_model, config.d_model)
                .with_bias(false)
                .with_layout(linear_layout)
                .init(device),
            rope,
            rotary_layout,
            num_query_heads: config.num_query_heads,
            num_kv_heads: config.num_kv_heads,
            head_dimension,
        }
    }

    fn project(
        &self,
        input: Tensor<B, 3>,
        position: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, sequence, _] = input.dims();
        let query = self
            .q_proj
            .forward(input.clone())
            .reshape([batch, sequence, self.num_query_heads, self.head_dimension])
            .swap_dims(1, 2);
        let key = self
            .k_proj
            .forward(input.clone())
            .reshape([batch, sequence, self.num_kv_heads, self.head_dimension])
            .swap_dims(1, 2);
        let value = self
            .v_proj
            .forward(input)
            .reshape([batch, sequence, self.num_kv_heads, self.head_dimension])
            .swap_dims(1, 2);
        (
            self.apply_rope(query, position),
            self.apply_rope(key, position),
            value,
        )
    }

    fn apply_rope(&self, input: Tensor<B, 4>, start: usize) -> Tensor<B, 4> {
        match self.rotary_layout {
            RotaryLayout::Interleaved => self.rope.apply(input, start),
            RotaryLayout::HalfSplit => apply_half_split_rope(&self.rope, input, start),
        }
    }

    fn finish(&self, context: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, _, sequence, _] = context.dims();
        let context = if sequence == 1 {
            // `[B, Hq, 1, D]` already has the same physical element order as
            // `[B, 1, Hq * D]`; swapping the two singleton-adjacent axes first
            // makes Ruda materialize an otherwise unnecessary copy.
            context.reshape([batch, sequence, self.num_query_heads * self.head_dimension])
        } else {
            context.swap_dims(1, 2).reshape([
                batch,
                sequence,
                self.num_query_heads * self.head_dimension,
            ])
        };
        self.o_proj.forward(context)
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let (query, key, value) = self.project(input, 0);
        let key = repeat_kv(key, self.num_query_heads / self.num_kv_heads);
        let value = repeat_kv(value, self.num_query_heads / self.num_kv_heads);
        self.finish(causal_attention(query, key, value, 0))
    }

    fn forward_cached(
        &self,
        input: Tensor<B, 3>,
        cache: &mut LlamaLayerCache<B>,
        position: usize,
    ) -> Tensor<B, 3> {
        assert_eq!(
            cache.sequence_length(),
            position,
            "all Llama layer caches must advance in lockstep"
        );
        let (query, key, value) = self.project(input, position);
        let (key, value) = cache.append(key, value);
        let key = repeat_kv(key, self.num_query_heads / self.num_kv_heads);
        let value = repeat_kv(value, self.num_query_heads / self.num_kv_heads);
        self.finish(causal_attention(query, key, value, position))
    }
}


#[derive(Module, Debug)]
pub struct LlamaFeedForward<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> LlamaFeedForward<B> {
    fn init(config: &LlamaConfig, device: &B::Device) -> Self {
        Self::init_with_layout(config, LinearLayout::Row, device)
    }

    fn init_qwen2(config: &LlamaConfig, device: &B::Device) -> Self {
        Self::init_with_layout(config, LinearLayout::Col, device)
    }

    fn init_with_layout(
        config: &LlamaConfig,
        linear_layout: LinearLayout,
        device: &B::Device,
    ) -> Self {
        Self {
            gate_proj: LinearConfig::new(config.d_model, config.d_ff)
                .with_bias(false)
                .with_layout(linear_layout)
                .init(device),
            up_proj: LinearConfig::new(config.d_model, config.d_ff)
                .with_bias(false)
                .with_layout(linear_layout)
                .init(device),
            down_proj: LinearConfig::new(config.d_ff, config.d_model)
                .with_bias(false)
                .with_layout(linear_layout)
                .init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.down_proj
            .forward(silu(self.gate_proj.forward(input.clone())) * self.up_proj.forward(input))
    }
}

#[derive(Module, Debug)]
pub struct LlamaDecoderLayer<B: Backend> {
    pub self_attn: LlamaAttention<B>,
    pub mlp: LlamaFeedForward<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> LlamaDecoderLayer<B> {
    fn init(config: &LlamaConfig, device: &B::Device) -> Self {
        Self::init_with_attention(
            config,
            LlamaAttention::init(config, device),
            LlamaFeedForward::init(config, device),
            device,
        )
    }

    fn init_qwen2_with_rope(
        config: &LlamaConfig,
        rope: RotaryEncoding<B>,
        device: &B::Device,
    ) -> Self {
        Self::init_with_attention(
            config,
            LlamaAttention::init_qwen2_with_rope(config, rope, device),
            LlamaFeedForward::init_qwen2(config, device),
            device,
        )
    }

    fn init_with_attention(
        config: &LlamaConfig,
        self_attn: LlamaAttention<B>,
        mlp: LlamaFeedForward<B>,
        device: &B::Device,
    ) -> Self {
        Self {
            self_attn,
            mlp,
            input_layernorm: RmsNormConfig::new(config.d_model)
                .with_epsilon(config.rms_norm_epsilon)
                .init(device),
            post_attention_layernorm: RmsNormConfig::new(config.d_model)
                .with_epsilon(config.rms_norm_epsilon)
                .init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = input.clone();
        let hidden = residual + self.self_attn.forward(self.input_layernorm.forward(input));
        let residual = hidden.clone();
        residual
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden))
    }

    fn forward_cached(
        &self,
        input: Tensor<B, 3>,
        cache: &mut LlamaLayerCache<B>,
        position: usize,
    ) -> Tensor<B, 3> {
        let residual = input.clone();
        let hidden = residual
            + self
                .self_attn
                .forward_cached(self.input_layernorm.forward(input), cache, position);
        let residual = hidden.clone();
        residual
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden))
    }
}


/// Decoder-only causal language model assembled from Ruda's native modules.
#[derive(Module, Debug)]
pub struct LlamaForCausalLm<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<LlamaDecoderLayer<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
    max_sequence_length: usize,
}

/// Llama/Qwen layout with trainable projection weights packed once after
/// loading. Q/K/V and gate/up each execute as one linear operation instead of
/// launching three and two independent GEMMs for every decoder layer.
#[derive(Module, Debug)]
pub struct PackedLlamaForCausalLm<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<PackedLlamaDecoderLayer<B>>,
    norm: RmsNorm<B>,
    lm_head: Linear<B>,
    max_sequence_length: usize,
}

#[derive(Module, Debug)]
struct PackedLlamaDecoderLayer<B: Backend> {
    self_attn: PackedLlamaAttention<B>,
    mlp: PackedLlamaFeedForward<B>,
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
}

#[derive(Module, Debug)]
struct PackedLlamaAttention<B: Backend> {
    qkv_weight: Param<Tensor<B, 2>>,
    qkv_bias: Option<Param<Tensor<B, 1>>>,
    o_proj: Linear<B>,
    rope: RotaryEncoding<B>,
    #[module(skip)]
    rotary_layout: RotaryLayout,
    num_query_heads: usize,
    num_kv_heads: usize,
    head_dimension: usize,
}

#[derive(Module, Debug)]
struct PackedLlamaFeedForward<B: Backend> {
    gate_up_weight: Param<Tensor<B, 2>>,
    down_proj: Linear<B>,
    d_ff: usize,
}


impl<B: Backend> LlamaForCausalLm<B> {
    pub fn init(config: &LlamaConfig, device: &B::Device) -> Result<Self, LlamaConfigError> {
        Self::init_with_layers(config, device, LlamaDecoderLayer::init, LinearLayout::Row)
    }

    /// Initialize the Qwen2/Qwen2.5 decoder variant. It shares the
    /// RMSNorm/SwiGLU/GQA skeleton with Llama, while retaining Qwen's Q/K/V
    /// biases and half-split rotary layout.
    pub fn init_qwen2(config: &LlamaConfig, device: &B::Device) -> Result<Self, LlamaConfigError> {
        config.validate()?;
        // Every decoder layer uses the same immutable RoPE frequencies. A
        // cloned Ruda tensor is a shared device handle, so build the cache
        // once instead of recomputing and storing one copy per layer.
        let rope = qwen2_rope::<B>(config, device);
        Self::init_with_layers(
            config,
            device,
            |config, device| LlamaDecoderLayer::init_qwen2_with_rope(config, rope.clone(), device),
            LinearLayout::Col,
        )
    }

    fn init_with_layers<F>(
        config: &LlamaConfig,
        device: &B::Device,
        mut init_layer: F,
        lm_head_layout: LinearLayout,
    ) -> Result<Self, LlamaConfigError>
    where
        F: FnMut(&LlamaConfig, &B::Device) -> LlamaDecoderLayer<B>,
    {
        config.validate()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.d_model).init(device),
            layers: (0..config.num_hidden_layers)
                .map(|_| init_layer(config, device))
                .collect(),
            norm: RmsNormConfig::new(config.d_model)
                .with_epsilon(config.rms_norm_epsilon)
                .init(device),
            lm_head: LinearConfig::new(config.d_model, config.vocab_size)
                .with_bias(false)
                .with_layout(lm_head_layout)
                .init(device),
            max_sequence_length: config.max_sequence_length,
        })
    }

    pub fn new_cache(&self) -> LlamaKvCache<B> {
        LlamaKvCache::new(self.layers.len(), self.max_sequence_length)
    }

    /// Full causal prefill. Returns logits in `[batch, sequence, vocabulary]` order.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let sequence = tokens.dims()[1];
        assert!(
            sequence <= self.max_sequence_length,
            "input sequence exceeds configured RoPE capacity"
        );
        let mut hidden = self.embed_tokens.forward(tokens);
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        self.lm_head.forward(self.norm.forward(hidden))
    }

    /// Appends one or more tokens to a KV cache and returns logits for only those tokens.
    pub fn forward_cached(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut LlamaKvCache<B>,
    ) -> Tensor<B, 3> {
        let hidden = self.forward_cached_hidden(tokens, cache);
        self.lm_head.forward(self.norm.forward(hidden))
    }

    /// Cached prefill/decode that projects only the final sequence position
    /// into vocabulary logits. Autoregressive generation never consumes the
    /// earlier prompt logits, so this avoids a large redundant LM-head GEMM.
    pub fn forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut LlamaKvCache<B>,
    ) -> Tensor<B, 3> {
        let hidden = self.forward_cached_hidden(tokens, cache);
        let [batch, sequence, width] = hidden.dims();
        assert!(
            sequence > 0,
            "last-logit forward requires at least one token"
        );
        let hidden = hidden.slice([0..batch, sequence - 1..sequence, 0..width]);
        self.lm_head.forward(self.norm.forward(hidden))
    }

    fn forward_cached_hidden(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut LlamaKvCache<B>,
    ) -> Tensor<B, 3> {
        assert_eq!(
            cache.layers.len(),
            self.layers.len(),
            "cache layer count does not match the model"
        );
        let sequence = tokens.dims()[1];
        let end = cache
            .position
            .checked_add(sequence)
            .expect("cache position overflow");
        assert!(
            end <= cache.max_sequence_length && end <= self.max_sequence_length,
            "KV cache capacity exceeded"
        );
        let mut hidden = self.embed_tokens.forward(tokens);
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden = layer.forward_cached(hidden, layer_cache, cache.position);
        }
        cache.position = end;
        hidden
    }

    /// Consume a loaded model and pack projection weights for low-overhead
    /// inference. Checkpoint loading remains unchanged and therefore retains
    /// exact Hugging Face tensor names and validation.
    pub fn into_packed_inference(self) -> PackedLlamaForCausalLm<B> {
        self.into_packed_training()
    }

    /// Consume a loaded model and pack Q/K/V and gate/up into trainable
    /// parameters. The returned module remains fully visible to Ruda's
    /// autodiff and optimizer traversal while reducing projection launches.
    pub fn into_packed_training(self) -> PackedLlamaForCausalLm<B> {
        let layers = self
            .layers
            .into_iter()
            .map(PackedLlamaDecoderLayer::from_layer)
            .collect();
        PackedLlamaForCausalLm {
            embed_tokens: self.embed_tokens,
            layers,
            norm: self.norm,
            lm_head: self.lm_head,
            max_sequence_length: self.max_sequence_length,
        }
    }
}

impl<B: Backend> PackedLlamaAttention<B> {
    fn from_attention(attention: LlamaAttention<B>) -> Self {
        let LlamaAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            rotary_layout,
            num_query_heads,
            num_kv_heads,
            head_dimension,
        } = attention;
        // Preserve the col-major view used by Qwen's LinearLayout::Col.
        // Ruda has optimized decode-time vec-mat kernels for this layout,
        // while a directly concatenated row-major RHS takes its slower path.
        let qkv_weight = Param::from_tensor(
            Tensor::cat(
                vec![
                    q_proj.weight.into_value().transpose(),
                    k_proj.weight.into_value().transpose(),
                    v_proj.weight.into_value().transpose(),
                ],
                0,
            )
            .transpose()
            .detach(),
        );
        let qkv_bias = match (q_proj.bias, k_proj.bias, v_proj.bias) {
            (Some(query), Some(key), Some(value)) => Some(Param::from_tensor(
                Tensor::cat(
                    vec![query.into_value(), key.into_value(), value.into_value()],
                    0,
                )
                .detach(),
            )),
            (None, None, None) => None,
            _ => panic!("Q/K/V projections must use the same bias layout"),
        };
        Self {
            qkv_weight,
            qkv_bias,
            o_proj,
            rope,
            rotary_layout,
            num_query_heads,
            num_kv_heads,
            head_dimension,
        }
    }

    fn apply_rope(&self, input: Tensor<B, 4>, start: usize) -> Tensor<B, 4> {
        match self.rotary_layout {
            RotaryLayout::Interleaved => self.rope.apply(input, start),
            RotaryLayout::HalfSplit => apply_half_split_rope(&self.rope, input, start),
        }
    }

    #[allow(clippy::single_range_in_vec_init)]
    fn forward(&self, input: Tensor<B, 3>, causal_mask: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        let [batch, sequence, input_width] = input.dims();
        let query_width = self.num_query_heads * self.head_dimension;
        let kv_width = self.num_kv_heads * self.head_dimension;
        let (query, key, value) = if batch * sequence < PACKED_TRAINING_MIN_TOKENS {
            let weight = self.qkv_weight.val();
            let bias = self.qkv_bias.as_ref().map(Param::val);
            let query = linear(
                input.clone(),
                weight.clone().slice([0..input_width, 0..query_width]),
                bias.clone().map(|bias| bias.slice([0..query_width])),
            );
            let key = linear(
                input.clone(),
                weight
                    .clone()
                    .slice([0..input_width, query_width..query_width + kv_width]),
                bias.clone()
                    .map(|bias| bias.slice([query_width..query_width + kv_width])),
            );
            let value = linear(
                input,
                weight.slice([
                    0..input_width,
                    query_width + kv_width..query_width + 2 * kv_width,
                ]),
                bias.map(|bias| bias.slice([query_width + kv_width..query_width + 2 * kv_width])),
            );
            (query, key, value)
        } else {
            let projected = linear(
                input,
                self.qkv_weight.val(),
                self.qkv_bias.as_ref().map(Param::val),
            );
            let query = projected
                .clone()
                .slice([0..batch, 0..sequence, 0..query_width]);
            let key = projected.clone().slice([
                0..batch,
                0..sequence,
                query_width..query_width + kv_width,
            ]);
            let value = projected.slice([
                0..batch,
                0..sequence,
                query_width + kv_width..query_width + 2 * kv_width,
            ]);
            (query, key, value)
        };
        let query = query
            .reshape([batch, sequence, self.num_query_heads, self.head_dimension])
            .swap_dims(1, 2);
        let key = key
            .reshape([batch, sequence, self.num_kv_heads, self.head_dimension])
            .swap_dims(1, 2);
        let value = value
            .reshape([batch, sequence, self.num_kv_heads, self.head_dimension])
            .swap_dims(1, 2);
        let query = self.apply_rope(query, 0);
        let key = self.apply_rope(key, 0);
        let key = repeat_kv(key, self.num_query_heads / self.num_kv_heads);
        let value = repeat_kv(value, self.num_query_heads / self.num_kv_heads);
        let context = match causal_mask {
            Some(mask) => causal_attention_with_mask(query, key, value, mask),
            None => attention(
                query,
                key,
                value,
                None,
                None,
                AttentionModuleOptions {
                    is_causal: true,
                    ..Default::default()
                },
            ),
        };
        self.o_proj.forward(context.swap_dims(1, 2).reshape([
            batch,
            sequence,
            self.num_query_heads * self.head_dimension,
        ]))
    }

    fn forward_cached(
        &self,
        input: Tensor<B, 3>,
        cache: &mut LlamaLayerCache<B>,
        position: usize,
    ) -> Tensor<B, 3> {
        assert_eq!(
            cache.sequence_length(),
            position,
            "all packed Llama layer caches must advance in lockstep"
        );
        let [batch, sequence, _] = input.dims();
        let query_width = self.num_query_heads * self.head_dimension;
        let kv_width = self.num_kv_heads * self.head_dimension;
        let projected = linear(
            input,
            self.qkv_weight.val(),
            self.qkv_bias.as_ref().map(Param::val),
        );
        let query = projected
            .clone()
            .slice([0..batch, 0..sequence, 0..query_width])
            .reshape([batch, sequence, self.num_query_heads, self.head_dimension])
            .swap_dims(1, 2);
        let key = projected
            .clone()
            .slice([0..batch, 0..sequence, query_width..query_width + kv_width])
            .reshape([batch, sequence, self.num_kv_heads, self.head_dimension])
            .swap_dims(1, 2);
        let value = projected
            .slice([
                0..batch,
                0..sequence,
                query_width + kv_width..query_width + 2 * kv_width,
            ])
            .reshape([batch, sequence, self.num_kv_heads, self.head_dimension])
            .swap_dims(1, 2);
        let query = self.apply_rope(query, position);
        let key = self.apply_rope(key, position);
        let (key, value) = cache.append(key, value);
        let key = repeat_kv(key, self.num_query_heads / self.num_kv_heads);
        let value = repeat_kv(value, self.num_query_heads / self.num_kv_heads);
        let context = causal_attention(query, key, value, position);
        self.o_proj.forward(context.swap_dims(1, 2).reshape([
            batch,
            sequence,
            self.num_query_heads * self.head_dimension,
        ]))
    }
}

impl<B: Backend> PackedLlamaFeedForward<B> {
    fn from_feed_forward(feed_forward: LlamaFeedForward<B>) -> Self {
        let LlamaFeedForward {
            gate_proj,
            up_proj,
            down_proj,
        } = feed_forward;
        let d_ff = gate_proj.weight.dims()[1];
        assert_eq!(
            up_proj.weight.dims()[1],
            d_ff,
            "gate and up projections must have equal widths"
        );
        let gate_up_weight = Param::from_tensor(
            Tensor::cat(
                vec![
                    gate_proj.weight.into_value().transpose(),
                    up_proj.weight.into_value().transpose(),
                ],
                0,
            )
            .transpose()
            .detach(),
        );
        Self {
            gate_up_weight,
            down_proj,
            d_ff,
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, sequence, input_width] = input.dims();
        let (gate, up) = if batch * sequence < PACKED_TRAINING_MIN_TOKENS {
            let weight = self.gate_up_weight.val();
            let gate = linear(
                input.clone(),
                weight.clone().slice([0..input_width, 0..self.d_ff]),
                None,
            );
            let up = linear(
                input,
                weight.slice([0..input_width, self.d_ff..2 * self.d_ff]),
                None,
            );
            (gate, up)
        } else {
            let gate_up = linear(input, self.gate_up_weight.val(), None);
            let gate = gate_up.clone().slice([0..batch, 0..sequence, 0..self.d_ff]);
            let up = gate_up.slice([0..batch, 0..sequence, self.d_ff..2 * self.d_ff]);
            (gate, up)
        };
        self.down_proj.forward(silu(gate) * up)
    }
}

impl<R, F, I, BT> PackedLlamaFeedForward<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn forward_ruda(
        &self,
        input: Tensor<DeviceBackend<R, F, I, BT>, 3>,
    ) -> Tensor<DeviceBackend<R, F, I, BT>, 3> {
        let gate_up = linear(input, self.gate_up_weight.val(), None);
        self.down_proj
            .forward(ruda_inference::swiglu(gate_up, self.d_ff))
    }
}

impl<R, F, I, BT> PackedLlamaAttention<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn forward_cached_ruda(
        &self,
        input: Tensor<DeviceBackend<R, F, I, BT>, 3>,
        cache: &mut LlamaLayerCache<DeviceBackend<R, F, I, BT>>,
        position: usize,
    ) -> Tensor<DeviceBackend<R, F, I, BT>, 3> {
        assert_eq!(
            cache.sequence_length(),
            position,
            "all packed Llama layer caches must advance in lockstep"
        );
        let [batch, sequence, _] = input.dims();
        let query_width = self.num_query_heads * self.head_dimension;
        let kv_width = self.num_kv_heads * self.head_dimension;
        let projected = linear(
            input,
            self.qkv_weight.val(),
            self.qkv_bias.as_ref().map(Param::val),
        );
        let (query, key, value, key_sequence) = match self.rotary_layout {
            RotaryLayout::HalfSplit => {
                let device = projected.device();
                let dtype = projected.dtype();
                let (key_storage, value_storage, cache_start, cache_end) = cache.reserve_growable(
                    batch,
                    self.num_kv_heads,
                    sequence,
                    self.head_dimension,
                    &device,
                    dtype,
                );
                let (query, key, value) = ruda_inference::qkv_half_split_rope_cached(
                    projected,
                    self.rope.freq_complex.clone(),
                    key_storage,
                    value_storage,
                    self.num_query_heads,
                    self.num_kv_heads,
                    self.head_dimension,
                    position,
                    cache_start,
                );
                let (key, value, key_sequence) = cache.commit_growable(key, value, cache_end);
                (query, key, value, key_sequence)
            }
            RotaryLayout::Interleaved => {
                let query = projected
                    .clone()
                    .slice([0..batch, 0..sequence, 0..query_width])
                    .reshape([batch, sequence, self.num_query_heads, self.head_dimension])
                    .swap_dims(1, 2);
                let key = projected
                    .clone()
                    .slice([0..batch, 0..sequence, query_width..query_width + kv_width])
                    .reshape([batch, sequence, self.num_kv_heads, self.head_dimension])
                    .swap_dims(1, 2);
                let value = projected
                    .slice([
                        0..batch,
                        0..sequence,
                        query_width + kv_width..query_width + 2 * kv_width,
                    ])
                    .reshape([batch, sequence, self.num_kv_heads, self.head_dimension])
                    .swap_dims(1, 2);
                let query = self.apply_rope(query, position);
                let key = self.apply_rope(key, position);
                let (key, value, key_sequence) = cache.append_growable(key, value);
                (query, key, value, key_sequence)
            }
        };
        let context = if sequence == 1 && self.head_dimension == 64 {
            ruda_inference::gqa_decode_attention(query, key, value, key_sequence)
        } else {
            let key = key.slice([
                0..batch,
                0..self.num_kv_heads,
                0..key_sequence,
                0..self.head_dimension,
            ]);
            let value = value.slice([
                0..batch,
                0..self.num_kv_heads,
                0..key_sequence,
                0..self.head_dimension,
            ]);
            let key = repeat_kv(key, self.num_query_heads / self.num_kv_heads);
            let value = repeat_kv(value, self.num_query_heads / self.num_kv_heads);
            causal_attention(query, key, value, position)
        };
        let context = if sequence == 1 {
            context.reshape([batch, sequence, self.num_query_heads * self.head_dimension])
        } else {
            context.swap_dims(1, 2).reshape([
                batch,
                sequence,
                self.num_query_heads * self.head_dimension,
            ])
        };
        self.o_proj.forward(context)
    }
}

impl<B: Backend> PackedLlamaDecoderLayer<B> {
    fn from_layer(layer: LlamaDecoderLayer<B>) -> Self {
        Self {
            self_attn: PackedLlamaAttention::from_attention(layer.self_attn),
            mlp: PackedLlamaFeedForward::from_feed_forward(layer.mlp),
            input_layernorm: layer.input_layernorm,
            post_attention_layernorm: layer.post_attention_layernorm,
        }
    }

    fn forward(&self, input: Tensor<B, 3>, causal_mask: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        let residual = input.clone();
        let hidden = residual
            + self
                .self_attn
                .forward(self.input_layernorm.forward(input), causal_mask);
        let residual = hidden.clone();
        residual
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden))
    }

    fn forward_cached(
        &self,
        input: Tensor<B, 3>,
        cache: &mut LlamaLayerCache<B>,
        position: usize,
    ) -> Tensor<B, 3> {
        let residual = input.clone();
        let hidden = residual
            + self
                .self_attn
                .forward_cached(self.input_layernorm.forward(input), cache, position);
        let residual = hidden.clone();
        residual
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden))
    }
}

impl<R, F, I, BT> PackedLlamaDecoderLayer<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn forward_cached_ruda(
        &self,
        input: Tensor<DeviceBackend<R, F, I, BT>, 3>,
        cache: &mut LlamaLayerCache<DeviceBackend<R, F, I, BT>>,
        position: usize,
    ) -> Tensor<DeviceBackend<R, F, I, BT>, 3> {
        let residual = input.clone();
        let normalized = ruda_inference::rms_norm(
            input,
            self.input_layernorm.gamma.val(),
            self.input_layernorm.epsilon,
        );
        let attention = self
            .self_attn
            .forward_cached_ruda(normalized, cache, position);
        let (hidden, normalized) = ruda_inference::residual_rms_norm(
            residual,
            attention,
            self.post_attention_layernorm.gamma.val(),
            self.post_attention_layernorm.epsilon,
        );
        hidden + self.mlp.forward_ruda(normalized)
    }
}

impl<B: Backend> PackedLlamaForCausalLm<B> {
    pub fn new_cache(&self) -> LlamaKvCache<B> {
        LlamaKvCache::new(self.layers.len(), self.max_sequence_length)
    }

    /// Full causal forward for pretraining and fine-tuning.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let sequence = tokens.dims()[1];
        assert!(
            sequence <= self.max_sequence_length,
            "input sequence exceeds configured RoPE capacity"
        );
        let causal_mask = (sequence < FLASH_ATTENTION_MIN_SEQUENCE)
            .then(|| causal_attention_mask(sequence, sequence, 0, &tokens.device()));
        let mut hidden = self.embed_tokens.forward(tokens);
        for layer in &self.layers {
            hidden = layer.forward(hidden, causal_mask.clone());
        }
        self.lm_head.forward(self.norm.forward(hidden))
    }

    pub fn forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut LlamaKvCache<B>,
    ) -> Tensor<B, 3> {
        assert_eq!(
            cache.layers.len(),
            self.layers.len(),
            "cache layer count does not match the packed model"
        );
        let sequence = tokens.dims()[1];
        let end = cache
            .position
            .checked_add(sequence)
            .expect("cache position overflow");
        assert!(
            end <= cache.max_sequence_length && end <= self.max_sequence_length,
            "KV cache capacity exceeded"
        );
        let mut hidden = self.embed_tokens.forward(tokens);
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden = layer.forward_cached(hidden, layer_cache, cache.position);
        }
        cache.position = end;
        let [batch, sequence, width] = hidden.dims();
        let hidden = hidden.slice([0..batch, sequence - 1..sequence, 0..width]);
        self.lm_head.forward(self.norm.forward(hidden))
    }
}

impl<R, F, I, BT> PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    /// Cached inference using the dedicated Ruda RMSNorm and SwiGLU kernels.
    pub fn forward_cached_last_ruda(
        &self,
        tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>,
        cache: &mut LlamaKvCache<DeviceBackend<R, F, I, BT>>,
    ) -> Tensor<DeviceBackend<R, F, I, BT>, 3> {
        assert_eq!(
            cache.layers.len(),
            self.layers.len(),
            "cache layer count does not match the packed model"
        );
        let sequence = tokens.dims()[1];
        let end = cache
            .position
            .checked_add(sequence)
            .expect("cache position overflow");
        assert!(
            end <= cache.max_sequence_length && end <= self.max_sequence_length,
            "KV cache capacity exceeded"
        );
        let mut hidden = self.embed_tokens.forward(tokens);
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden = layer.forward_cached_ruda(hidden, layer_cache, cache.position);
        }
        cache.position = end;
        let [batch, sequence, width] = hidden.dims();
        let hidden = hidden.slice([0..batch, sequence - 1..sequence, 0..width]);
        let hidden = ruda_inference::rms_norm(hidden, self.norm.gamma.val(), self.norm.epsilon);
        self.lm_head.forward(hidden)
    }
}



/// Qwen2 repeats each RoPE frequency across the first and second halves of a
/// head, then rotates those halves against one another. Ruda's stock
/// `RotaryEncoding` instead rotates adjacent pairs, so recover the cached
/// cosine/sine values and apply Qwen's exact layout explicitly.
fn apply_half_split_rope<B: Backend>(
    rope: &RotaryEncoding<B>,
    input: Tensor<B, 4>,
    start: usize,
) -> Tensor<B, 4> {
    let [_, _, sequence, dimension] = input.dims();
    let maximum = rope.freq_complex.dims()[0];
    let end = start
        .checked_add(sequence)
        .expect("Qwen RoPE position overflow");
    assert!(end <= maximum, "Qwen RoPE position exceeds cache");
    assert_eq!(dimension % 2, 0, "Qwen RoPE head dimension must be even");
    let half = dimension / 2;

    // `freq_complex` is `[position, half, duplicate, cos_sin]` after
    // reshaping; retain one duplicate before constructing Qwen's
    // `[freqs, freqs]` layout.
    let pairs = rope
        .freq_complex
        .clone()
        .slice([start..end, 0..dimension, 0..2])
        .reshape([sequence, half, 2, 2])
        .slice([0..sequence, 0..half, 0..1, 0..2])
        .reshape([sequence, half, 2]);
    let cosine_half = pairs
        .clone()
        .slice([0..sequence, 0..half, 0..1])
        .reshape([sequence, half]);
    let sine_half = pairs
        .slice([0..sequence, 0..half, 1..2])
        .reshape([sequence, half]);
    let cosine =
        Tensor::cat(vec![cosine_half.clone(), cosine_half], 1).reshape([1, 1, sequence, dimension]);
    let sine =
        Tensor::cat(vec![sine_half.clone(), sine_half], 1).reshape([1, 1, sequence, dimension]);

    let first = input
        .clone()
        .slice([0..input.dims()[0], 0..input.dims()[1], 0..sequence, 0..half]);
    let second = input.clone().slice([
        0..input.dims()[0],
        0..input.dims()[1],
        0..sequence,
        half..dimension,
    ]);
    let rotated = Tensor::cat(vec![-second, first], 3);
    input * cosine + rotated * sine
}

fn repeat_kv<B: Backend>(input: Tensor<B, 4>, repeats: usize) -> Tensor<B, 4> {
    if repeats == 1 {
        return input;
    }
    let [batch, heads, sequence, dimension] = input.dims();
    input.unsqueeze_dim::<5>(2).repeat_dim(2, repeats).reshape([
        batch,
        heads * repeats,
        sequence,
        dimension,
    ])
}

fn causal_attention_mask<B: Backend>(
    query_sequence: usize,
    key_sequence: usize,
    query_start: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    assert!(query_start + query_sequence <= key_sequence);
    let mut mask = Vec::with_capacity(query_sequence * key_sequence);
    for query_index in 0..query_sequence {
        let last_visible_key = query_start + query_index;
        for key_index in 0..key_sequence {
            mask.push(if key_index <= last_visible_key {
                0.0_f32
            } else {
                -1.0e30_f32
            });
        }
    }
    Tensor::<B, 4>::from_data(
        TensorData::new(mask, [1, 1, query_sequence, key_sequence]),
        device,
    )
}

fn causal_attention_with_mask<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    mask: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [batch, query_heads, query_sequence, head_dimension] = query.dims();
    let [key_batch, key_heads, key_sequence, key_dimension] = key.dims();
    assert_eq!([batch, query_heads], [key_batch, key_heads]);
    assert_eq!(head_dimension, key_dimension);
    assert_eq!(
        value.dims(),
        [batch, query_heads, key_sequence, head_dimension]
    );
    assert_eq!(mask.dims(), [1, 1, query_sequence, key_sequence]);

    let scale = 1.0 / (head_dimension as f32).sqrt();
    let value_dtype = value.dtype();
    let mask = mask.cast(query.dtype());
    let scores = query.matmul(key.swap_dims(2, 3)).mul_scalar(scale) + mask;
    let probabilities = softmax(scores.cast(DType::F32), 3).cast(value_dtype);
    probabilities.matmul(value)
}

fn causal_attention<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    query_start: usize,
) -> Tensor<B, 4> {
    let [batch, heads, query_sequence, head_dimension] = query.dims();
    let [key_batch, key_heads, key_sequence, key_dimension] = key.dims();
    assert_eq!([batch, heads], [key_batch, key_heads]);
    assert_eq!(head_dimension, key_dimension);
    assert_eq!(value.dims(), [batch, heads, key_sequence, head_dimension]);
    assert!(query_start + query_sequence <= key_sequence);

    let device = query.device();
    let mask = causal_attention_mask(query_sequence, key_sequence, query_start, &device);
    // Qwen/Llama attention evaluates softmax in FP32 even when Q/K/V and the
    // returned context use a half dtype. This avoids accumulating the
    // normalization denominator at BF16/FP16 precision.
    causal_attention_with_mask(query, key, value, mask)
}

#[cfg(all(test, feature = "nvidia"))]
mod tests {
    use super::*;
    use ruda_tensor_device::cuda::CudaDevice;
    type TestBackend = ruda_tensor_device::cuda::Cuda<f32, i32>;

    fn tiny_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 16,
            d_model: 8,
            d_ff: 16,
            num_hidden_layers: 1,
            num_query_heads: 2,
            num_kv_heads: 1,
            max_sequence_length: 8,
            rms_norm_epsilon: 1.0e-6,
            rope_theta: 10_000.0,
        }
    }

    #[cfg(feature = "nvidia")]
    #[test]
    fn qwen_bf16_rope_preserves_position_precision() {
        use half::bf16;
        let device = ruda_tensor_device::cuda::CudaDevice::default();
        let mut config = tiny_config();
        config.max_sequence_length = 1026;
        let rope = rope::qwen2_rope_with_dtype::<ruda_tensor_device::cuda::Cuda<bf16, i32>>(
            &config, &device, DType::BF16,
        );
        let actual = rope.freq_complex
            .into_data().to_vec::<bf16>().unwrap();
        let dimension = config.head_dimension();
        for position in [0, 1, 255, 256, 257, 1025] {
            for pair in 0..dimension / 2 {
                let angle = position as f32 / (config.rope_theta as f32).powf((2 * pair) as f32 / dimension as f32);
                let index = (position * dimension + 2 * pair) * 2;
                assert_eq!(actual[index], bf16::from_f32(angle.cos()), "cos {position}/{pair}");
                assert_eq!(actual[index + 1], bf16::from_f32(angle.sin()), "sin {position}/{pair}");
            }
        }
    }

    #[test]
    fn qwen_initialization_has_qkv_biases() {
        let device = CudaDevice::default();
        let model = LlamaForCausalLm::<TestBackend>::init_qwen2(&tiny_config(), &device).unwrap();
        let attention = &model.layers[0].self_attn;
        assert!(attention.q_proj.bias.is_some());
        assert!(attention.k_proj.bias.is_some());
        assert!(attention.v_proj.bias.is_some());
        assert!(attention.o_proj.bias.is_none());
        assert_eq!(attention.rotary_layout, RotaryLayout::HalfSplit);
    }

    #[test]
    fn qwen_half_split_rope_matches_reference_equations() {
        let device = CudaDevice::default();
        let mut config = tiny_config();
        config.max_sequence_length = 4;
        let rope = rope::qwen2_rope_with_dtype::<TestBackend>(&config, &device, DType::F32);
        let input = Tensor::<TestBackend, 4>::from_data([[[[1.0, 2.0, 3.0, 4.0]]]], (&device, DType::F32));
        let actual = apply_half_split_rope(&rope, input, 1)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let cos0 = 1.0_f32.cos();
        let sin0 = 1.0_f32.sin();
        let cos1 = 0.01_f32.cos();
        let sin1 = 0.01_f32.sin();
        let expected = [
            cos0 - 3.0 * sin0,
            2.0 * cos1 - 4.0 * sin1,
            3.0 * cos0 + sin0,
            4.0 * cos1 + 2.0 * sin1,
        ];
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 2.0e-3, "{actual} != {expected}");
        }
    }

    #[test]
    fn packed_training_forward_matches_standard_forward() {
        struct F32Parameters;
        impl ruda_model::module::ModuleMapper<TestBackend> for F32Parameters {
            fn map_float<const D: usize>(
                &mut self,
                param: Param<Tensor<TestBackend, D>>,
            ) -> Param<Tensor<TestBackend, D>> {
                param.map(|tensor| tensor.cast(DType::F32))
            }
        }
        let device = CudaDevice::default();
        let config = tiny_config();
        let mut model = LlamaForCausalLm::<TestBackend>::init_qwen2(&config, &device)
            .unwrap().map(&mut F32Parameters);
        let rope = rope::qwen2_rope_with_dtype::<TestBackend>(&config, &device, DType::F32);
        for layer in &mut model.layers {
            layer.self_attn.rope = rope.clone();
        }
        let tokens = Tensor::<TestBackend, 2, Int>::from_data([[1, 2, 3, 4]], &device);
        fn compare<const D: usize>(label: &str, a: Tensor<TestBackend, D>, b: Tensor<TestBackend, D>) {
            assert_eq!(a.dtype(), b.dtype(), "{label} dtype");
            let a = a.into_data().to_vec::<f32>().unwrap();
            let b = b.into_data().to_vec::<f32>().unwrap();
            for (index, (a, b)) in a.into_iter().zip(b).enumerate() {
                assert!((a - b).abs() < 3.0e-3, "{label}[{index}]: {a} != {b}");
            }
        }
        let packed = model.clone().into_packed_training();
        let layer = &model.layers[0];
        let packed_layer = &packed.layers[0];
        compare("q weight", layer.self_attn.q_proj.weight.val(),
            packed_layer.self_attn.qkv_weight.val().slice([0..8, 0..8]));
        compare("k weight", layer.self_attn.k_proj.weight.val(),
            packed_layer.self_attn.qkv_weight.val().slice([0..8, 8..12]));
        compare("v weight", layer.self_attn.v_proj.weight.val(),
            packed_layer.self_attn.qkv_weight.val().slice([0..8, 12..16]));
        let hidden = model.embed_tokens.forward(tokens.clone());
        let normalized = layer.input_layernorm.forward(hidden.clone());
        compare("attention", layer.self_attn.forward(normalized.clone()),
            packed_layer.self_attn.forward(normalized.clone(),
                Some(causal_attention_mask(4, 4, 0, &device))));
        compare("mlp", layer.mlp.forward(normalized.clone()), packed_layer.mlp.forward(normalized));
        compare("layer", layer.forward(hidden.clone()), packed_layer.forward(hidden,
            Some(causal_attention_mask(4, 4, 0, &device))));
        let expected = model
            .forward(tokens.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let actual = model
            .into_packed_training()
            .forward(tokens)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 3.0e-3, "{actual} != {expected}");
        }
    }
}
