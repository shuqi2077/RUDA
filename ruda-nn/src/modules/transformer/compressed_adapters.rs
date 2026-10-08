use alloc::vec::Vec;
use ruda_model::{module::{Initializer, Module}, tensor::{Tensor, backend::Backend}};
use crate::{Linear, LinearConfig, Embedding, Dropout, DropoutConfig,
    attention::{CompressedAttentionProjection, CompressedAttentionProjectionRole}};
use super::{AdaptedProjection, TransformerAdapterConfig, MhcTransformerBlock,
    HybridAttentionBackbone, HybridAttentionHead, HybridAttentionLanguageModel};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MhcTransformerProjectionRole {
    Attention(CompressedAttentionProjectionRole), Gate, Up, Down,
}

/// Exact native layer/role selection, independent of checkpoint/model-family names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridAttentionAdapterTarget {
    Layer(usize, MhcTransformerProjectionRole),
    Head,
}

/// A/B update for the existing row-major embedding head, without another base parameter.
#[derive(Module, Debug)]
pub struct HybridTiedEmbeddingAdapter<B: Backend> {
    pub adapter_a: Linear<B>,
    pub adapter_b: Linear<B>,
    pub dropout: Dropout,
    pub scale: f64,
}

impl<B: Backend> HybridTiedEmbeddingAdapter<B> {
    /// Attach original loaded adapter leaves. Freeze the complete shared embedding
    /// explicitly first; no transposed base copy or new embedding ID is created.
    pub fn from_adapters(embedding: &Embedding<B>, adapter_a: Linear<B>, adapter_b: Linear<B>, dropout: Dropout, scale: f64) -> Self {
        let adapter = Self { adapter_a, adapter_b, dropout, scale };
        adapter.validate(embedding);
        adapter
    }

    pub fn init(embedding: &Embedding<B>, config: &TransformerAdapterConfig) -> Self {
        check_options(config);
        let weight = embedding.weight.val();
        assert!(!weight.is_require_grad(), "freeze the complete shared embedding before attaching its head adapter");
        let [vocab, width] = weight.dims();
        let rank = config.lora.rank;
        let device = weight.device();
        let dtype = config.adapter_dtype.unwrap_or(weight.dtype());
        let mut a: Linear<B> = LinearConfig::new(width, rank).with_bias(false).init(&device);
        let mut b: Linear<B> = LinearConfig::new(rank, vocab).with_bias(false).with_initializer(Initializer::Zeros).init(&device);
        a.weight = a.weight.map(|value| value.cast(dtype).detach().require_grad());
        b.weight = b.weight.map(|value| value.cast(dtype).detach().require_grad());
        let denominator = if config.use_rslora { (rank as f64).sqrt() } else { rank as f64 };
        Self::from_adapters(embedding, a, b, DropoutConfig::new(config.lora.dropout).init(), config.lora.alpha / denominator)
    }

    pub(crate) fn validate(&self, embedding: &Embedding<B>) {
        let weight = embedding.weight.val();
        let [vocab, width] = weight.dims();
        let a = self.adapter_a.weight.val();
        let b = self.adapter_b.weight.val();
        let rank = a.dims()[1];
        assert!(vocab > 0 && width > 0 && rank > 0 && weight.dtype().is_float()
            && !weight.is_require_grad(), "tied head adapter requires actual nonempty frozen floating embedding storage");
        assert_eq!(a.dims(), [width, rank], "tied head A width/rank differs");
        assert_eq!(b.dims(), [rank, vocab], "tied head B rank/vocabulary differs");
        assert!(self.adapter_a.bias.is_none() && self.adapter_b.bias.is_none(), "tied head adapters must be bias-free");
        assert!(a.device() == weight.device() && b.device() == weight.device() && a.dtype().is_float() && b.dtype().is_float(),
            "tied head adapter device/storage differs");
        assert!(self.scale.is_finite() && self.dropout.prob.is_finite() && (0.0..1.0).contains(&self.dropout.prob),
            "invalid tied head adapter scale/dropout");
    }

    pub(crate) fn forward(&self, hidden: Tensor<B, 3>, embedding: &Embedding<B>) -> Tensor<B, 3> {
        let base = hidden.clone().matmul(embedding.weight.val().transpose().unsqueeze::<3>());
        let adapted = self.dropout.forward(hidden.cast(self.adapter_a.weight.val().dtype()));
        let update = self.adapter_b.forward(self.adapter_a.forward(adapted).cast(self.adapter_b.weight.val().dtype())).mul_scalar(self.scale);
        let storage = base.dtype();
        base + update.cast(storage)
    }
}

fn check_options(config: &TransformerAdapterConfig) {
    assert!(config.lora.rank > 0 && config.lora.alpha.is_finite(), "invalid hybrid adapter rank/alpha");
    assert!(config.lora.dropout.is_finite() && (0.0..1.0).contains(&config.lora.dropout), "invalid hybrid adapter dropout");
    assert!(config.adapter_dtype.is_none_or(|dtype| dtype.is_float()), "hybrid adapter storage must be floating");
}

fn unique<T: PartialEq>(targets: &[T]) {
    for (index, target) in targets.iter().enumerate() { assert!(!targets[..index].contains(target), "duplicate native hybrid adapter target"); }
}

fn wrap<B: Backend>(projection: Linear<B>, selected: bool, config: &TransformerAdapterConfig) -> AdaptedProjection<B> {
    if selected {
        let dtype = config.adapter_dtype.unwrap_or_else(|| projection.weight.val().dtype());
        AdaptedProjection::LoRA(config.lora.init_with_options(projection, dtype, config.use_rslora))
    } else { AdaptedProjection::Dense(projection) }
}

impl<B: Backend, P: CompressedAttentionProjection<B>> MhcTransformerBlock<B, P> {
    pub fn visit_projections<'a>(&'a self, mut visitor: impl FnMut(MhcTransformerProjectionRole, &'a P)) {
        self.attention.visit_projections(|role, projection| visitor(MhcTransformerProjectionRole::Attention(role), projection));
        visitor(MhcTransformerProjectionRole::Gate, &self.gate);
        visitor(MhcTransformerProjectionRole::Up, &self.up);
        visitor(MhcTransformerProjectionRole::Down, &self.down);
    }

    pub fn map_projections<Q: CompressedAttentionProjection<B>>(self,
        mut mapper: impl FnMut(MhcTransformerProjectionRole, P) -> Q) -> MhcTransformerBlock<B, Q> {
        match self.try_map_projections(|role, projection| Ok::<Q, core::convert::Infallible>(mapper(role, projection))) {
            Ok(module) => module, Err(error) => match error {},
        }
    }

    pub fn try_map_projections<Q: CompressedAttentionProjection<B>, E>(self,
        mut mapper: impl FnMut(MhcTransformerProjectionRole, P) -> Result<Q, E>) -> Result<MhcTransformerBlock<B, Q>, E> {
        let attention = self.attention.try_map_projections(|role, projection| mapper(MhcTransformerProjectionRole::Attention(role), projection))?;
        let gate = mapper(MhcTransformerProjectionRole::Gate, self.gate)?;
        let up = mapper(MhcTransformerProjectionRole::Up, self.up)?;
        let down = mapper(MhcTransformerProjectionRole::Down, self.down)?;
        Ok(MhcTransformerBlock::from_parts(self.attention_connection, self.ffn_connection, attention,
            self.attention_norm, self.ffn_norm, gate, up, down, self.epsilon))
    }
}

impl<B: Backend> MhcTransformerBlock<B> {
    /// Attach real LoRA/rsLoRA only to explicitly selected original roles, including
    /// compressor/indexer channels. All mHC/norm/sink leaves retain their existing flags.
    pub fn with_adapters(self, config: &TransformerAdapterConfig, targets: &[MhcTransformerProjectionRole])
        -> MhcTransformerBlock<B, AdaptedProjection<B>> {
        unique(targets);
        if !targets.is_empty() { check_options(config); }
        let mut actual = Vec::new();
        self.visit_projections(|role, _| actual.push(role));
        assert!(targets.iter().all(|target| actual.contains(target)), "selected hybrid block projection does not exist");
        self.map_projections(|role, projection| wrap(projection, targets.contains(&role), config))
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>> HybridAttentionBackbone<B, P> {
    pub fn visit_projections<'a>(&'a self, mut visitor: impl FnMut(HybridAttentionAdapterTarget, &'a P)) {
        for (layer, block) in self.layers.iter().enumerate() {
            block.visit_projections(|role, projection| visitor(HybridAttentionAdapterTarget::Layer(layer, role), projection));
        }
    }

    pub fn map_projections<Q: CompressedAttentionProjection<B>>(self,
        mut mapper: impl FnMut(HybridAttentionAdapterTarget, P) -> Q) -> HybridAttentionBackbone<B, Q> {
        match self.try_map_projections(|role, projection| Ok::<Q, core::convert::Infallible>(mapper(role, projection))) {
            Ok(module) => module, Err(error) => match error {},
        }
    }

    pub fn try_map_projections<Q: CompressedAttentionProjection<B>, E>(self,
        mut mapper: impl FnMut(HybridAttentionAdapterTarget, P) -> Result<Q, E>) -> Result<HybridAttentionBackbone<B, Q>, E> {
        let layers = self.layers.into_iter().enumerate().map(|(layer, block)| block.try_map_projections(|role, projection|
            mapper(HybridAttentionAdapterTarget::Layer(layer, role), projection))).collect::<Result<_, E>>()?;
        Ok(HybridAttentionBackbone::from_parts(self.embedding, layers, self.final_norm, self.epsilon))
    }
}

impl<B: Backend> HybridAttentionBackbone<B> {
    /// Validate the complete declared target set before allocating any A/B matrices.
    pub fn with_adapters(self, config: &TransformerAdapterConfig, targets: &[HybridAttentionAdapterTarget])
        -> HybridAttentionBackbone<B, AdaptedProjection<B>> {
        unique(targets);
        if !targets.is_empty() { check_options(config); }
        let mut actual = Vec::new();
        self.visit_projections(|target, _| actual.push(target));
        assert!(targets.iter().all(|target| actual.contains(target)), "selected hybrid backbone projection does not exist");
        self.map_projections(|target, projection| wrap(projection, targets.contains(&target), config))
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>> HybridAttentionLanguageModel<B, P> {
    /// Real owned projection modules; tied-head base/update leaves are kept separate
    /// rather than manufactured as another dense parameter with a conflicting layout.
    pub fn visit_projections<'a>(&'a self, mut visitor: impl FnMut(HybridAttentionAdapterTarget, &'a P)) {
        self.backbone.visit_projections(&mut visitor);
        if let HybridAttentionHead::Linear(head) = &self.head { visitor(HybridAttentionAdapterTarget::Head, head); }
    }

    pub fn map_projections<Q: CompressedAttentionProjection<B>>(self,
        mut mapper: impl FnMut(HybridAttentionAdapterTarget, P) -> Q) -> HybridAttentionLanguageModel<B, Q> {
        match self.try_map_projections(|role, projection| Ok::<Q, core::convert::Infallible>(mapper(role, projection))) {
            Ok(module) => module, Err(error) => match error {},
        }
    }

    pub fn try_map_projections<Q: CompressedAttentionProjection<B>, E>(self,
        mut mapper: impl FnMut(HybridAttentionAdapterTarget, P) -> Result<Q, E>) -> Result<HybridAttentionLanguageModel<B, Q>, E> {
        let backbone = self.backbone.try_map_projections(&mut mapper)?;
        let head = match self.head {
            HybridAttentionHead::Linear(head) => HybridAttentionHead::Linear(mapper(HybridAttentionAdapterTarget::Head, head)?),
            HybridAttentionHead::TiedEmbedding(_) => HybridAttentionHead::TiedEmbedding(core::marker::PhantomData),
            HybridAttentionHead::TiedEmbeddingLoRA(adapter) => HybridAttentionHead::TiedEmbeddingLoRA(adapter),
        };
        Ok(HybridAttentionLanguageModel::from_parts(backbone, head))
    }
}

impl<B: Backend> HybridAttentionLanguageModel<B> {
    /// Real native model adaptation. Untied selected bases use RUDA's existing LoRA;
    /// a selected tied head adds only A/B to the already-frozen shared embedding.
    pub fn with_adapters(self, config: &TransformerAdapterConfig, targets: &[HybridAttentionAdapterTarget])
        -> HybridAttentionLanguageModel<B, AdaptedProjection<B>> {
        unique(targets);
        if !targets.is_empty() { check_options(config); }
        let mut actual = Vec::new();
        self.visit_projections(|target, _| actual.push(target));
        actual.push(HybridAttentionAdapterTarget::Head);
        assert!(targets.iter().all(|target| actual.contains(target)), "selected hybrid model projection does not exist");
        let selected_head = targets.contains(&HybridAttentionAdapterTarget::Head);
        if selected_head {
            match &self.head {
                HybridAttentionHead::TiedEmbedding(_) => assert!(!self.backbone.embedding.weight.val().is_require_grad(),
                    "freeze the complete shared embedding before selecting its head adapter"),
                HybridAttentionHead::TiedEmbeddingLoRA(_) => panic!("selected tied head already has an adapter"),
                HybridAttentionHead::Linear(_) => {}
            }
        }
        let backbone = self.backbone.map_projections(|target, projection| wrap(projection, targets.contains(&target), config));
        let head = match self.head {
            HybridAttentionHead::Linear(head) => HybridAttentionHead::Linear(wrap(head, selected_head, config)),
            HybridAttentionHead::TiedEmbedding(_) if selected_head => HybridAttentionHead::TiedEmbeddingLoRA(
                HybridTiedEmbeddingAdapter::init(&backbone.embedding, config)),
            HybridAttentionHead::TiedEmbedding(_) => HybridAttentionHead::TiedEmbedding(core::marker::PhantomData),
            HybridAttentionHead::TiedEmbeddingLoRA(adapter) => HybridAttentionHead::TiedEmbeddingLoRA(adapter),
        };
        HybridAttentionLanguageModel::from_parts(backbone, head)
    }
}
