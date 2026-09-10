mod attention;
mod loading;
mod projection;

use super::*;
use projection::AwqProjection;

pub type AwqBackend<R> = DeviceBackend<R, half::f16, i32, u32>;

#[derive(Debug)]
pub struct AwqLlamaForCausalLm<R: DeviceRuntime>
where
    R::Device: DeviceOps,
{
    embed_tokens: Embedding<AwqBackend<R>>,
    layers: Vec<AwqDecoderLayer<R>>,
    norm: RmsNorm<AwqBackend<R>>,
    lm_head: AwqProjection<R>,
    max_sequence_length: usize,
}

#[derive(Debug)]
struct AwqDecoderLayer<R: DeviceRuntime>
where
    R::Device: DeviceOps,
{
    self_attn: attention::AwqAttention<R>,
    mlp: AwqFeedForward<R>,
    input_layernorm: RmsNorm<AwqBackend<R>>,
    post_attention_layernorm: RmsNorm<AwqBackend<R>>,
}

#[derive(Debug)]
struct AwqFeedForward<R: DeviceRuntime>
where
    R::Device: DeviceOps,
{
    gate_proj: AwqProjection<R>,
    up_proj: AwqProjection<R>,
    down_proj: AwqProjection<R>,
}

impl<R: DeviceRuntime> AwqFeedForward<R>
where
    R::Device: DeviceOps,
{
    fn forward(&self, input: Tensor<AwqBackend<R>, 3>) -> Tensor<AwqBackend<R>, 3> {
        self.down_proj
            .forward(silu(self.gate_proj.forward(input.clone())) * self.up_proj.forward(input))
    }
}

impl<R: DeviceRuntime> AwqDecoderLayer<R>
where
    R::Device: DeviceOps,
{
    fn forward(&self, input: Tensor<AwqBackend<R>, 3>) -> Tensor<AwqBackend<R>, 3> {
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
        input: Tensor<AwqBackend<R>, 3>,
        cache: &mut LlamaLayerCache<AwqBackend<R>>,
        position: usize,
    ) -> Tensor<AwqBackend<R>, 3> {
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

impl<R: DeviceRuntime> AwqLlamaForCausalLm<R>
where
    R::Device: DeviceOps,
{
    pub fn new_cache(&self) -> LlamaKvCache<AwqBackend<R>> {
        LlamaKvCache::new(self.layers.len(), self.max_sequence_length)
    }

    /// Full causal prefill. Returns logits in `[batch, sequence, vocabulary]` order.
    pub fn forward(&self, tokens: Tensor<AwqBackend<R>, 2, Int>) -> Tensor<AwqBackend<R>, 3> {
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
        tokens: Tensor<AwqBackend<R>, 2, Int>,
        cache: &mut LlamaKvCache<AwqBackend<R>>,
    ) -> Tensor<AwqBackend<R>, 3> {
        let hidden = self.forward_cached_hidden(tokens, cache);
        self.lm_head.forward(self.norm.forward(hidden))
    }

    /// Cached prefill/decode that projects only the final sequence position
    /// into vocabulary logits. Autoregressive generation never consumes the
    /// earlier prompt logits, so this avoids a large redundant LM-head GEMM.
    pub fn forward_cached_last(
        &self,
        tokens: Tensor<AwqBackend<R>, 2, Int>,
        cache: &mut LlamaKvCache<AwqBackend<R>>,
    ) -> Tensor<AwqBackend<R>, 3> {
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
        tokens: Tensor<AwqBackend<R>, 2, Int>,
        cache: &mut LlamaKvCache<AwqBackend<R>>,
    ) -> Tensor<AwqBackend<R>, 3> {
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
}
