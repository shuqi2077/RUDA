use alloc::{collections::BTreeMap,vec::Vec};
use ruda_model::{config::Config,module::Module,tensor::{Tensor,backend::Backend}};
use crate::{Dropout,attention::{DenseAttentionMask,DenseAttentionOptions}};
use super::{DenseCrossAttentionBlock,DenseEncoderDecoderLayer,DenseEncoderDecoderStack,DenseTransformerNorm,
    AdaptedGroupedQueryAttention,AdaptedTransformerBlock,AdaptedStackLayer,TransformerAdapterConfig,
    AttentionAdapterTarget,FeedForwardAdapterTarget};
use super::dense::residual_branch;

/// Independently configured native self-attention and FFN adapter targets.
#[derive(Config,Debug)]
pub struct DecoderBackboneAdapterConfig {
    /// Actual LoRA/rsLoRA rank, alpha, dropout and A/B storage for this backbone.
    pub adapter: TransformerAdapterConfig,
    /// Explicit self-attention projections, independent of cross-attention.
    pub attention: Vec<AttentionAdapterTarget>,
    /// Explicit FFN projections; an absent gate cannot be selected.
    pub feed_forward: Vec<FeedForwardAdapterTarget>,
}

/// Independently configured native encoder-memory attention adapter targets.
#[derive(Config,Debug)]
pub struct CrossAttentionAdapterConfig {
    /// Rank, alpha, dropout, dtype and scaling convention for memory attention only.
    pub adapter: TransformerAdapterConfig,
    /// Exact Q/K/V/output projections; actual memory and query widths may differ.
    pub attention: Vec<AttentionAdapterTarget>,
}

/// Explicit adapter configuration for one actual encoder-decoder layer.
#[derive(Config,Debug)]
pub struct DecoderLayerAdapterConfig {
    /// Zero-based index of the supplied layer, not a model-name pattern.
    pub layer: usize,
    /// None preserves the original dense self-attention and FFN without conversion.
    pub backbone: Option<DecoderBackboneAdapterConfig>,
    /// None preserves the original dense encoder-memory stage without conversion.
    pub cross_attention: Option<CrossAttentionAdapterConfig>,
}

fn check_config(config: &TransformerAdapterConfig) {
    assert!(config.lora.rank > 0 && config.lora.alpha.is_finite(),"invalid decoder adapter rank/alpha");
    assert!(config.lora.dropout.is_finite() && (0.0..1.0).contains(&config.lora.dropout),"invalid decoder adapter dropout");
    assert!(config.adapter_dtype.is_none_or(|dtype|dtype.is_float()),"decoder adapter storage must be floating");
}

fn check_targets<T: PartialEq>(targets: &[T]) {
    for (index,target) in targets.iter().enumerate() {
        assert!(!targets[..index].contains(target),"duplicate decoder adapter projection target");
    }
}

impl DecoderLayerAdapterConfig {
    fn validate_for<B: Backend>(&self,base: &DenseEncoderDecoderLayer<B>) {
        assert!(self.backbone.is_some() || self.cross_attention.is_some(),"selected decoder layer needs an actual adapter stage");
        if let Some(config) = &self.backbone {
            check_config(&config.adapter);
            check_targets(&config.attention);
            check_targets(&config.feed_forward);
            assert!(!config.attention.is_empty() || !config.feed_forward.is_empty(),"decoder backbone needs an actual adapter target");
            assert!(base.backbone.feed_forward.gate.is_some() || !config.feed_forward.contains(&FeedForwardAdapterTarget::Gate),
                "decoder backbone has no selected gate projection");
        }
        if let Some(config) = &self.cross_attention {
            check_config(&config.adapter);
            check_targets(&config.attention);
            assert!(!config.attention.is_empty(),"cross-attention needs an actual adapter projection");
        }
    }
}

/// Encoder-memory attention with adapters on explicitly selected actual projections.
#[derive(Module,Debug)]
pub struct AdaptedCrossAttentionBlock<B: Backend> {
    /// Original query/memory weights with adapters only on requested projections.
    pub attention: AdaptedGroupedQueryAttention<B>,
    /// Original query/residual normalization, including its parameter IDs and flags.
    pub query_norm: DenseTransformerNorm<B>,
    /// Optional original memory normalization; it is neither added nor discarded.
    pub memory_norm: Option<DenseTransformerNorm<B>>,
    /// Original branch-output dropout.
    pub residual_dropout: Dropout,
    /// Original pre/post normalization order.
    pub norm_first: bool,
}

impl<B: Backend> AdaptedCrossAttentionBlock<B> {
    /// Add only selected native A/B projections, retaining the actual memory geometry.
    /// Unselected parameters remain unchanged; freeze the base explicitly for A/B-only training.
    pub fn from_dense(base: DenseCrossAttentionBlock<B>,config: &CrossAttentionAdapterConfig) -> Self {
        check_config(&config.adapter);
        check_targets(&config.attention);
        assert!(!config.attention.is_empty(),"cross-attention needs an actual adapter projection");
        Self {attention:AdaptedGroupedQueryAttention::from_dense(base.attention,&config.adapter,&config.attention),
            query_norm:base.query_norm,memory_norm:base.memory_norm,residual_dropout:base.residual_dropout,norm_first:base.norm_first}
    }

    /// Native cross-attention with independent caller-supplied query/memory visibility.
    pub fn forward(&self,input: Tensor<B,3>,memory: Tensor<B,3>,masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_with_positions(input,memory,masks,options,|query,key|(query,key))
    }

    /// Transform actual projected query and encoder-memory key positions independently.
    pub fn forward_with_positions<F>(&self,input: Tensor<B,3>,memory: Tensor<B,3>,masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let memory = if let Some(norm) = &self.memory_norm { norm.forward(memory) } else { memory };
        residual_branch(input,&self.query_norm,&self.residual_dropout,self.norm_first,|source| {
            let (query,key,value) = self.attention.project(source,memory.clone(),memory);
            let geometry = (query.dims(),key.dims());
            let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"adapted cross positions changed head geometry");
            self.attention.forward_projected(query,key,value,masks,options)
        })
    }
}

/// Original dense or explicitly adapted encoder-memory attention stage.
#[derive(Module,Debug)]
pub enum DecoderCrossAttention<B: Backend> {
    /// Actual original dense stage and all its parameter identities.
    Dense(DenseCrossAttentionBlock<B>),
    /// Actual stage with explicitly selected native LoRA/rsLoRA projections.
    Adapted(AdaptedCrossAttentionBlock<B>),
}

impl<B: Backend> DecoderCrossAttention<B> {
    /// Execute the actual dense/adapted memory attention without implicit masks.
    pub fn forward(&self,input: Tensor<B,3>,memory: Tensor<B,3>,masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_with_positions(input,memory,masks,options,|query,key|(query,key))
    }

    /// Preserve caller-owned query/key position transforms on either stage type.
    pub fn forward_with_positions<F>(&self,input: Tensor<B,3>,memory: Tensor<B,3>,masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        match self {
            Self::Dense(block)=>block.forward_with_positions(input,memory,masks,options,positions),
            Self::Adapted(block)=>block.forward_with_positions(input,memory,masks,options,positions),
        }
    }
}

/// Native fine-tuning layer preserving self-attention, memory attention, then FFN.
#[derive(Module,Debug)]
pub struct AdaptedEncoderDecoderLayer<B: Backend> {
    /// Actual dense/adapted self-attention and FFN stages with independent norms.
    pub backbone: AdaptedStackLayer<B>,
    /// Actual dense/adapted encoder-memory stage inserted before the FFN.
    pub cross_attention: DecoderCrossAttention<B>,
}

impl<B: Backend> AdaptedEncoderDecoderLayer<B> {
    /// Convert only explicitly selected stages; self/cross configurations are independent.
    /// For adapter-only training, freeze the original layer before this conversion.
    pub fn from_dense(base: DenseEncoderDecoderLayer<B>,config: &DecoderLayerAdapterConfig) -> Self {
        config.validate_for(&base);
        let backbone = if let Some(config) = &config.backbone {
            AdaptedStackLayer::Adapted(AdaptedTransformerBlock::from_dense(base.backbone,&config.adapter,
                &config.attention,&config.feed_forward))
        } else { AdaptedStackLayer::Dense(base.backbone) };
        let cross_attention = if let Some(config) = &config.cross_attention {
            DecoderCrossAttention::Adapted(AdaptedCrossAttentionBlock::from_dense(base.cross_attention,config))
        } else { DecoderCrossAttention::Dense(base.cross_attention) };
        Self {backbone,cross_attention}
    }

    /// Retain every original parameter, with neither adapter allocation nor blanket freezing.
    pub fn dense(base: DenseEncoderDecoderLayer<B>) -> Self {
        Self {backbone:AdaptedStackLayer::Dense(base.backbone),cross_attention:DecoderCrossAttention::Dense(base.cross_attention)}
    }

    /// Forward with distinct actual self-attention and encoder-memory visibility rules.
    pub fn forward(&self,input: Tensor<B,3>,memory: Tensor<B,3>,self_masks: DenseAttentionMask<B>,
        self_options: DenseAttentionOptions,memory_masks: DenseAttentionMask<B>,memory_options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_with_positions(input,memory,self_masks,self_options,memory_masks,memory_options,
            |query,key|(query,key),|query,key|(query,key))
    }

    /// Native self/cross positional transforms without inferred offsets or model family.
    pub fn forward_with_positions<F,G>(&self,input: Tensor<B,3>,memory: Tensor<B,3>,
        self_masks: DenseAttentionMask<B>,self_options: DenseAttentionOptions,memory_masks: DenseAttentionMask<B>,
        memory_options: DenseAttentionOptions,self_positions: F,cross_positions: G) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>),
        G: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden = self.backbone.forward_attention_with_positions(input,self_masks,self_options,self_positions);
        let hidden = self.cross_attention.forward_with_positions(hidden,memory,memory_masks,memory_options,cross_positions);
        self.backbone.forward_feed_forward(hidden)
    }
}

/// Layer-selected native encoder-decoder fine-tuning with independent cross-attention adapters.
#[derive(Module,Debug)]
pub struct AdaptedEncoderDecoderStack<B: Backend> {
    /// Every actual decoder layer, including unchanged dense layers, in original order.
    pub layers: Vec<AdaptedEncoderDecoderLayer<B>>,
}

impl<B: Backend> AdaptedEncoderDecoderStack<B> {
    /// Validate every actual layer/target before allocating any selected A/B matrices.
    /// Unselected layers and stages retain their existing values, identities and flags.
    pub fn from_dense(base: DenseEncoderDecoderStack<B>,targets: &[DecoderLayerAdapterConfig]) -> Self {
        let mut selected = BTreeMap::new();
        for target in targets {
            assert!(target.layer < base.layers.len(),"decoder adapter layer index is outside the actual stack");
            assert!(selected.insert(target.layer,target).is_none(),"duplicate decoder adapter layer index");
            target.validate_for(&base.layers[target.layer]);
        }
        let layers = base.layers.into_iter().enumerate().map(|(index,layer)| {
            if let Some(config) = selected.get(&index) { AdaptedEncoderDecoderLayer::from_dense(layer,config) }
            else { AdaptedEncoderDecoderLayer::dense(layer) }
        }).collect();
        Self {layers}
    }

    /// Connect actual already-prepared layers without resetting their parameter state.
    pub fn new(layers: Vec<AdaptedEncoderDecoderLayer<B>>) -> Self { Self {layers} }

    /// Apply explicit shared self/cross masks/options to every original/adapted layer.
    pub fn forward(&self,mut input: Tensor<B,3>,memory: Tensor<B,3>,self_masks: DenseAttentionMask<B>,
        self_options: DenseAttentionOptions,memory_masks: DenseAttentionMask<B>,memory_options: DenseAttentionOptions) -> Tensor<B,3> {
        for layer in &self.layers {
            input = layer.forward(input,memory.clone(),self_masks.clone(),self_options,memory_masks.clone(),memory_options);
        }
        input
    }

    /// Caller-owned per-layer positions/visibility/configuration with unchanged layer order.
    pub fn forward_with<F>(&self,mut input: Tensor<B,3>,memory: Tensor<B,3>,mut layer: F) -> Tensor<B,3>
    where F: FnMut(usize,&AdaptedEncoderDecoderLayer<B>,Tensor<B,3>,Tensor<B,3>)->Tensor<B,3> {
        for (index,block) in self.layers.iter().enumerate() { input = layer(index,block,input,memory.clone()); }
        input
    }
}
