use alloc::vec::Vec;
use ruda_model::{config::Config, module::{Initializer, Module, Param},
    tensor::{Bool, DType, FloatDType, Int, Tensor, backend::Backend}};
use crate::{Embedding, EmbeddingConfig, Linear, LinearConfig, Mhc, MhcConfig,
    attention::{CompressedAttention, CompressedAttentionConfig, CompressedAttentionOutput, CompressedAttentionSession, CompressedAttentionProjection}};
use super::HybridTiedEmbeddingAdapter;

pub(super) fn normalized<B: Backend>(input: Tensor<B, 3>, weight: &Param<Tensor<B, 1>>, epsilon: f64) -> Tensor<B, 3> {
    let storage = input.dtype();
    let compute = if storage == DType::F64 { DType::F64 } else { DType::F32 };
    let input = input.cast(compute);
    let denominator = (input.clone().square().mean_dim(2) + epsilon).sqrt();
    ((input / denominator) * weight.val().cast(compute).unsqueeze::<3>()).cast(storage)
}

pub(super) fn visible<B: Backend>(input: &Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 2, Bool> {
    let [batch, tokens, _] = input.dims();
    if let Some(valid) = valid {
        assert_eq!(valid.dims(), [batch, tokens], "hybrid visibility must describe the actual token slots");
        assert_eq!(valid.device(), input.device(), "hybrid visibility device differs");
        valid
    } else { Tensor::<B, 2, Bool>::zeros([batch, tokens], &input.device()).bool_not() }
}

/// Explicit compressed-attention block with two independent mHC residual connections.
#[derive(Config, Debug)]
pub struct MhcTransformerBlockConfig {
    pub attention: CompressedAttentionConfig,
    /// Actual gated FFN intermediate width, not a model-family expansion heuristic.
    pub feedforward_width: usize,
    #[config(default = 4)]
    pub streams: usize,
    #[config(default = 20)]
    pub sinkhorn_iterations: usize,
}

/// mHC attention and gated SiLU FFN, retaining stream state across both sublayers.
#[derive(Module, Debug)]
pub struct MhcTransformerBlock<B: Backend, P: Module<B> = Linear<B>> {
    pub attention_connection: Mhc<B>,
    pub ffn_connection: Mhc<B>,
    pub attention: CompressedAttention<B, P>,
    pub attention_norm: Param<Tensor<B, 1>>,
    pub ffn_norm: Param<Tensor<B, 1>>,
    pub gate: P,
    pub up: P,
    pub down: P,
    pub epsilon: f64,
}

#[derive(Clone, Debug)]
pub struct MhcTransformerOutput<B: Backend> {
    pub state: Tensor<B, 4>,
    pub indexer_loss: Tensor<B, 1>,
}

impl MhcTransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MhcTransformerBlock<B> {
        assert!(self.feedforward_width > 0, "hybrid feed-forward width must be positive");
        let width = self.attention.width;
        let epsilon = self.attention.epsilon;
        let connection = MhcConfig::new(width).with_streams(self.streams)
            .with_sinkhorn_iterations(self.sinkhorn_iterations).with_epsilon(epsilon);
        MhcTransformerBlock::from_parts(connection.init(device), connection.init(device), self.attention.init(device),
            Initializer::Ones.init([width], device), Initializer::Ones.init([width], device),
            LinearConfig::new(width, self.feedforward_width).with_bias(false).init(device),
            LinearConfig::new(width, self.feedforward_width).with_bias(false).init(device),
            LinearConfig::new(self.feedforward_width, width).with_bias(false).init(device), epsilon)
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>> MhcTransformerBlock<B, P> {
    /// Connect original loaded branches and two independent residual mappings without reinitialization.
    pub fn from_parts(attention_connection: Mhc<B>, ffn_connection: Mhc<B>, attention: CompressedAttention<B, P>,
        attention_norm: Param<Tensor<B, 1>>, ffn_norm: Param<Tensor<B, 1>>, gate: P, up: P,
        down: P, epsilon: f64) -> Self {
        let width = attention.width;
        assert!(epsilon.is_finite() && epsilon > 0.0, "invalid hybrid block epsilon");
        assert_eq!((attention_connection.width, ffn_connection.width), (width, width), "hybrid connection widths differ");
        assert_eq!(attention_connection.streams, ffn_connection.streams, "hybrid residual stream counts differ");
        assert_eq!(attention_norm.val().dims(), [width], "hybrid attention normalization width differs");
        assert_eq!(ffn_norm.val().dims(), [width], "hybrid FFN normalization width differs");
        let [up_input, hidden] = up.dimensions();
        assert!(hidden > 0, "hybrid FFN intermediate width must be positive");
        assert_eq!(up_input, width, "hybrid FFN input width differs");
        assert_eq!(gate.dimensions(), [width, hidden], "hybrid gate/value geometry differs");
        assert_eq!(down.dimensions(), [hidden, width], "hybrid FFN output geometry differs");
        let device = attention.parts.query_down.device();
        for projection in [&gate, &up, &down] {
            assert!(!projection.has_bias(), "hybrid gated FFN projections must be bias-free");
            assert_eq!(projection.device(), device, "hybrid FFN device differs");
        }
        assert!(attention_norm.val().device() == device && ffn_norm.val().device() == device
            && attention_connection.mapping.val().device() == device && ffn_connection.mapping.val().device() == device,
            "hybrid normalization/residual device differs");
        Self { attention_connection, ffn_connection, attention, attention_norm, ffn_norm, gate, up, down, epsilon }
    }

    fn finish(&self, state: Tensor<B, 4>, attention: Tensor<B, 3>, mappings: crate::MhcMappings<B>,
        valid: Tensor<B, 2, Bool>) -> Tensor<B, 4> {
        let state = self.attention_connection.post(state, attention, mappings);
        let (merged, mappings) = self.ffn_connection.pre(state.clone());
        let input = normalized(merged, &self.ffn_norm, self.epsilon);
        let update = self.down.forward(ruda_model::tensor::activation::silu(self.gate.forward(input.clone())) * self.up.forward(input));
        let [batch, tokens, _] = update.dims();
        let update = update.clone() * valid.cast::<FloatDType>(update.dtype().into()).reshape([batch, tokens, 1]);
        self.ffn_connection.post(state, update, mappings)
    }

    pub fn forward(&self, state: Tensor<B, 4>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 4> {
        let (query, mappings) = self.attention_connection.pre(state.clone());
        let normalized = normalized(query, &self.attention_norm, self.epsilon);
        let valid = visible(&normalized, valid);
        let attention = self.attention.forward(normalized, Some(valid.clone()));
        self.finish(state, attention, mappings, valid)
    }

    /// Training and independent indexer-only warm-up use the same actual mHC/FFN branches.
    pub fn forward_with_aux(&self, state: Tensor<B, 4>, valid: Option<Tensor<B, 2, Bool>>,
        indexer_warmup: bool) -> MhcTransformerOutput<B> {
        let (query, mappings) = self.attention_connection.pre(state.clone());
        let normalized = normalized(query, &self.attention_norm, self.epsilon);
        let valid = visible(&normalized, valid);
        let attention = self.attention.forward_with_aux(normalized, Some(valid.clone()), indexer_warmup);
        MhcTransformerOutput { state: self.finish(state, attention.output, mappings, valid), indexer_loss: attention.indexer_loss }
    }

    pub fn inference_session(&self) -> MhcTransformerSession<'_, B, P> {
        MhcTransformerSession { block: self, attention: self.attention.inference_session() }
    }
}

/// Incremental state bound to the unchanged attention, mHC and FFN parameters.
#[derive(Debug)]
pub struct MhcTransformerSession<'a, B: Backend, P: Module<B> = Linear<B>> {
    block: &'a MhcTransformerBlock<B, P>,
    attention: CompressedAttentionSession<'a, B, P>,
}

impl<'a, B: Backend, P: CompressedAttentionProjection<B>> MhcTransformerSession<'a, B, P> {
    pub fn position(&self) -> usize { self.attention.position() }
    pub fn clear(&mut self) { self.attention.clear(); }
    pub fn reorder(&mut self, parents: Tensor<B, 1, Int>) { self.attention.reorder(parents); }
    pub fn compressed_blocks(&self) -> usize { self.attention.compressed_blocks() }
    pub fn retained_raw_tokens(&self) -> usize { self.attention.retained_raw_tokens() }
    pub fn tensor_bytes(&self) -> usize { self.attention.tensor_bytes() }

    pub fn fork(&self) -> Self { Self { block: self.block, attention: self.attention.fork() } }

    pub fn restore(&mut self, snapshot: Self) {
        assert!(core::ptr::eq(self.block, snapshot.block), "hybrid block snapshot belongs to another module");
        self.attention.restore(snapshot.attention);
    }

    pub fn forward(&mut self, state: Tensor<B, 4>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 4> {
        let (query, mappings) = self.block.attention_connection.pre(state.clone());
        let normalized = normalized(query, &self.block.attention_norm, self.block.epsilon);
        let valid = visible(&normalized, valid);
        let attention = self.attention.forward(normalized, Some(valid.clone()));
        self.block.finish(state, attention, mappings, valid).detach()
    }
}

/// Caller-selected per-layer CSA/HCA geometry, without inferred large-model presets.
#[derive(Config, Debug)]
pub struct HybridAttentionBackboneConfig {
    pub vocab_size: usize,
    pub width: usize,
    pub layers: Vec<MhcTransformerBlockConfig>,
    #[config(default = 1e-6)]
    pub epsilon: f64,
}

/// Trainable embedding, explicit heterogeneous compressed/mHC blocks and final RMS weight.
#[derive(Module, Debug)]
pub struct HybridAttentionBackbone<B: Backend, P: Module<B> = Linear<B>> {
    pub embedding: Embedding<B>,
    pub layers: Vec<MhcTransformerBlock<B, P>>,
    pub final_norm: Param<Tensor<B, 1>>,
    pub epsilon: f64,
}

impl HybridAttentionBackboneConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HybridAttentionBackbone<B> {
        assert!(self.vocab_size > 0 && self.width > 0 && !self.layers.is_empty(), "hybrid backbone geometry must be nonempty");
        assert!(self.layers.iter().all(|layer| layer.attention.width == self.width), "hybrid layer residual widths differ");
        HybridAttentionBackbone::from_parts(EmbeddingConfig::new(self.vocab_size, self.width).init(device),
            self.layers.iter().map(|layer| layer.init(device)).collect(), Initializer::Ones.init([self.width], device), self.epsilon)
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>> HybridAttentionBackbone<B, P> {
    pub fn from_parts(embedding: Embedding<B>, layers: Vec<MhcTransformerBlock<B, P>>, final_norm: Param<Tensor<B, 1>>, epsilon: f64) -> Self {
        let [vocab, width] = embedding.weight.val().dims();
        assert!(vocab > 0 && width > 0 && !layers.is_empty() && epsilon.is_finite() && epsilon > 0.0,
            "invalid loaded hybrid backbone geometry/epsilon");
        let streams = layers[0].attention_connection.streams;
        assert_eq!(final_norm.val().dims(), [width], "hybrid final normalization width differs");
        let device = embedding.weight.val().device();
        assert_eq!(final_norm.val().device(), device, "hybrid final normalization device differs");
        for layer in &layers {
            assert_eq!((layer.attention.width, layer.attention_connection.streams, layer.ffn_connection.streams),
                (width, streams, streams), "hybrid layer width/stream geometry differs");
            assert_eq!(layer.attention.parts.query_down.device(), device, "hybrid layer device differs");
        }
        Self { embedding, layers, final_norm, epsilon }
    }

    fn embed(&self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>) -> (Tensor<B, 3>, Tensor<B, 2, Bool>) {
        let [batch, length] = tokens.dims();
        assert!(batch > 0 && length > 0 && matches!(tokens.dtype(), DType::I32 | DType::I64), "hybrid tokens must be nonempty I32/I64 rows");
        assert_eq!(tokens.device(), self.embedding.weight.val().device(), "hybrid tokens device differs");
        let hidden = self.embedding.forward(tokens);
        let valid = visible(&hidden, valid);
        let hidden = hidden.clone() * valid.clone().cast::<FloatDType>(hidden.dtype().into()).reshape([batch, length, 1]);
        (hidden, valid)
    }

    /// Normalized actual backbone features, without allocating full-vocabulary logits.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        let (hidden, valid) = self.embed(tokens, valid);
        let mut state = self.layers[0].attention_connection.expand(hidden);
        for layer in &self.layers { state = layer.forward(state, Some(valid.clone())); }
        normalized(self.layers.last().unwrap().ffn_connection.reduce(state), &self.final_norm, self.epsilon)
    }

    /// Sum the actual independent per-layer KL losses; HCA remains a disconnected zero.
    pub fn forward_with_aux(&self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>,
        indexer_warmup: bool) -> CompressedAttentionOutput<B> {
        let (hidden, valid) = self.embed(tokens, valid);
        let mut state = self.layers[0].attention_connection.expand(hidden);
        let mut losses = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let result = layer.forward_with_aux(state, Some(valid.clone()), indexer_warmup);
            state = result.state;
            losses.push(result.indexer_loss);
        }
        CompressedAttentionOutput {
            output: normalized(self.layers.last().unwrap().ffn_connection.reduce(state), &self.final_norm, self.epsilon),
            indexer_loss: Tensor::cat(losses, 0).sum(),
        }
    }

    pub fn inference_session(&self) -> HybridAttentionBackboneSession<'_, B, P> {
        HybridAttentionBackboneSession { backbone: self, layers: self.layers.iter().map(MhcTransformerBlock::inference_session).collect() }
    }
}

/// Exact same borrowed layer set for arbitrary-size prefill/decode chunks.
#[derive(Debug)]
pub struct HybridAttentionBackboneSession<'a, B: Backend, P: Module<B> = Linear<B>> {
    backbone: &'a HybridAttentionBackbone<B, P>,
    layers: Vec<MhcTransformerSession<'a, B, P>>,
}

impl<'a, B: Backend, P: CompressedAttentionProjection<B>> HybridAttentionBackboneSession<'a, B, P> {
    pub fn position(&self) -> usize {
        let position = self.layers[0].position();
        assert!(self.layers.iter().all(|layer| layer.position() == position), "hybrid layer cache positions differ");
        position
    }

    pub fn clear(&mut self) { for layer in &mut self.layers { layer.clear(); } }

    pub fn reorder(&mut self, parents: Tensor<B, 1, Int>) {
        for layer in &mut self.layers { layer.reorder(parents.clone()); }
    }

    pub fn compressed_blocks(&self) -> Vec<usize> { self.layers.iter().map(MhcTransformerSession::compressed_blocks).collect() }
    pub fn retained_raw_tokens(&self) -> Vec<usize> { self.layers.iter().map(MhcTransformerSession::retained_raw_tokens).collect() }
    pub fn tensor_bytes(&self) -> usize { self.layers.iter().map(MhcTransformerSession::tensor_bytes).sum() }

    /// Retain every actual layer's local/compressed/raw-tail state as one native snapshot.
    pub fn fork(&self) -> Self {
        self.position();
        Self { backbone: self.backbone, layers: self.layers.iter().map(MhcTransformerSession::fork).collect() }
    }

    pub fn restore(&mut self, snapshot: Self) {
        assert!(core::ptr::eq(self.backbone, snapshot.backbone), "hybrid backbone snapshot belongs to another module");
        snapshot.position();
        for (layer, saved) in self.layers.iter_mut().zip(snapshot.layers) { layer.restore(saved); }
    }

    pub fn forward(&mut self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        let start = self.position();
        let tokens_count = tokens.dims()[1];
        let (hidden, valid) = self.backbone.embed(tokens, valid);
        let mut state = self.backbone.layers[0].attention_connection.expand(hidden);
        for layer in &mut self.layers { state = layer.forward(state, Some(valid.clone())); }
        assert_eq!(self.position(), start.checked_add(tokens_count).expect("hybrid position overflow"));
        normalized(self.backbone.layers.last().unwrap().ffn_connection.reduce(state), &self.backbone.final_norm, self.backbone.epsilon).detach()
    }
}

/// Actual untied linear output or a view of the existing embedding leaf.
#[derive(Module, Debug)]
pub enum HybridAttentionHead<B: Backend, P: Module<B> = Linear<B>> {
    Linear(P),
    TiedEmbedding(core::marker::PhantomData<B>),
    TiedEmbeddingLoRA(HybridTiedEmbeddingAdapter<B>),
}

impl<B: Backend, P: CompressedAttentionProjection<B>> HybridAttentionHead<B, P> {
    pub(super) fn forward(&self, hidden: Tensor<B, 3>, embedding: &Embedding<B>) -> Tensor<B, 3> {
        match self {
            Self::Linear(head) => head.forward(hidden),
            Self::TiedEmbedding(_) => hidden.matmul(embedding.weight.val().transpose().unsqueeze::<3>()),
            Self::TiedEmbeddingLoRA(adapter) => adapter.forward(hidden, embedding),
        }
    }
}

/// Complete native generic compressed/mHC language model with optional true embedding tie.
#[derive(Module, Debug)]
pub struct HybridAttentionLanguageModel<B: Backend, P: Module<B> = Linear<B>> {
    pub backbone: HybridAttentionBackbone<B, P>,
    pub head: HybridAttentionHead<B, P>,
}

impl<B: Backend, P: CompressedAttentionProjection<B>> HybridAttentionLanguageModel<B, P> {
    pub fn from_parts(backbone: HybridAttentionBackbone<B, P>, head: HybridAttentionHead<B, P>) -> Self {
        if let HybridAttentionHead::Linear(head) = &head {
            let [vocab, width] = backbone.embedding.weight.val().dims();
            assert_eq!(head.dimensions(), [width, vocab], "hybrid language head geometry differs");
            assert_eq!(head.device(), backbone.embedding.weight.val().device(), "hybrid language head device differs");
        }
        if let HybridAttentionHead::TiedEmbeddingLoRA(adapter) = &head { adapter.validate(&backbone.embedding); }
        Self { backbone, head }
    }

    pub fn with_tied_embeddings(backbone: HybridAttentionBackbone<B, P>) -> Self {
        Self::from_parts(backbone, HybridAttentionHead::TiedEmbedding(core::marker::PhantomData))
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        self.head.forward(self.backbone.forward(tokens, valid), &self.backbone.embedding)
    }

    pub fn forward_with_aux(&self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>,
        indexer_warmup: bool) -> CompressedAttentionOutput<B> {
        let result = self.backbone.forward_with_aux(tokens, valid, indexer_warmup);
        CompressedAttentionOutput { output: self.head.forward(result.output, &self.backbone.embedding), indexer_loss: result.indexer_loss }
    }

    pub fn inference_session(&self) -> HybridAttentionLanguageSession<'_, B, P> {
        HybridAttentionLanguageSession { model: self, backbone: self.backbone.inference_session() }
    }
}

#[derive(Debug)]
pub struct HybridAttentionLanguageSession<'a, B: Backend, P: Module<B> = Linear<B>> {
    model: &'a HybridAttentionLanguageModel<B, P>,
    backbone: HybridAttentionBackboneSession<'a, B, P>,
}

impl<'a, B: Backend, P: CompressedAttentionProjection<B>> HybridAttentionLanguageSession<'a, B, P> {
    pub fn position(&self) -> usize { self.backbone.position() }
    pub fn clear(&mut self) { self.backbone.clear(); }
    pub fn reorder(&mut self, parents: Tensor<B, 1, Int>) { self.backbone.reorder(parents); }
    pub fn compressed_blocks(&self) -> Vec<usize> { self.backbone.compressed_blocks() }
    pub fn retained_raw_tokens(&self) -> Vec<usize> { self.backbone.retained_raw_tokens() }
    pub fn tensor_bytes(&self) -> usize { self.backbone.tensor_bytes() }

    pub fn fork(&self) -> Self { Self { model: self.model, backbone: self.backbone.fork() } }

    pub fn restore(&mut self, snapshot: Self) {
        assert!(core::ptr::eq(self.model, snapshot.model), "hybrid language snapshot belongs to another module/head");
        self.backbone.restore(snapshot.backbone);
    }

    pub fn forward_hidden(&mut self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        self.backbone.forward(tokens, valid)
    }

    pub fn forward(&mut self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        let hidden = self.forward_hidden(tokens, valid);
        self.model.head.forward(hidden, &self.model.backbone.embedding).detach()
    }

    /// Run every new token through all layers, but project only the final physical
    /// slot to vocabulary logits. Hidden history is not discarded from the caches.
    pub fn forward_last(&mut self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        let hidden = self.forward_hidden(tokens, valid);
        let length = hidden.dims()[1];
        self.model.head.forward(hidden.slice_dim(1, length - 1..length), &self.model.backbone.embedding).detach()
    }
}
