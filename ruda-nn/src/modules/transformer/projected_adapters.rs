use alloc::collections::BTreeMap;
use ruda_model::tensor::{DType,backend::Backend};
use crate::{Linear,attention::GroupedQueryAttention};
use super::{TransformerAdapterConfig,LayerAdapterConfig,AttentionAdapterTarget,FeedForwardAdapterTarget,
    TransformerProjectionShape,AwqTransformerProjection,Nf4TransformerProjection,MixedTransformerProjection,
    QuantizedTransformerProjection,UniversalTransformerProjection,
    AwqGroupedQueryAttention,AwqFeedForward,AwqTransformerBlock,AwqTransformerStack,AwqTransformerModel,
    DenseFeedForward,DenseTransformerBlock,DenseTransformerStack,TransformerEmbeddings,TransformerHead,DenseTransformerNorm,AdaptedProjection,
    ProjectedCrossAttentionBlock,ProjectedEncoderDecoderLayer,ProjectedEncoderDecoderStack,ProjectedEncoderDecoderModel,
    DenseCrossAttentionBlock,DenseEncoderDecoderLayer,DenseEncoderDecoderStack,DecoderLayerAdapterConfig};

/// Add adapters to an explicitly selected original projection without changing its base format.
pub trait AdaptTransformerProjection<B:Backend>:TransformerProjectionShape<B> {
    /// Validate before any selected A/B allocation; an already adapted role is not re-adapted.
    fn validate_adapter(&self,config:&TransformerAdapterConfig);
    /// Attach native A/B leaves only to this selected original projection.
    fn with_adapter(self,config:&TransformerAdapterConfig) -> Self;
}

fn validate_config(config:&TransformerAdapterConfig,packed:bool) {
    assert!(config.lora.rank>0 && config.lora.alpha.is_finite(),"invalid native projection adapter rank/alpha");
    assert!(config.lora.dropout.is_finite() && (0.0..1.0).contains(&config.lora.dropout),"invalid native projection adapter dropout");
    if packed {assert!(config.adapter_dtype.is_some(),"packed projection adapters require an explicit floating adapter_dtype");}
    if let Some(dtype)=config.adapter_dtype {
        assert!(if packed {matches!(dtype,DType::F16|DType::BF16|DType::F32)} else {dtype.is_float()},"unsupported native projection adapter storage");
    }
}
fn unique<T:PartialEq>(targets:&[T]) {
    for (index,target) in targets.iter().enumerate() {assert!(!targets[..index].contains(target),"duplicate native projection adapter target");}
}

macro_rules! adapt_original_projection {
    ($projection:ident,[$($base:ident=>$adapted:ident,$init:ident),*]) => {
        impl<B:Backend> From<Linear<B>> for $projection<B> {
            fn from(layer:Linear<B>) -> Self {Self::Dense(layer)}
        }
        impl<B:Backend> AdaptTransformerProjection<B> for $projection<B> {
            fn validate_adapter(&self,config:&TransformerAdapterConfig) {
                match self {
                    Self::Dense(_)=>validate_config(config,false),
                    $(Self::$base(layer)=>{validate_config(config,true);layer.validate();},)*
                    _=>panic!("selected native projection already has an adapter"),
                }
            }
            fn with_adapter(self,config:&TransformerAdapterConfig) -> Self {
                self.validate_adapter(config);
                match self {
                    Self::Dense(layer)=>{
                        let dtype=config.adapter_dtype.unwrap_or_else(||layer.weight.val().dtype());
                        Self::LoRA(config.lora.init_with_options(layer,dtype,config.use_rslora))
                    },
                    $(Self::$base(layer)=>Self::$adapted(config.lora.$init(layer,config.adapter_dtype.expect("validated packed adapter dtype"),config.use_rslora)),)*
                    _=>unreachable!("validated projection is not already adapted"),
                }
            }
        }
    };
}
adapt_original_projection!(AwqTransformerProjection,[Awq=>AwqLoRA,init_awq]);
adapt_original_projection!(Nf4TransformerProjection,[Nf4=>Nf4LoRA,init_nf4]);
adapt_original_projection!(MixedTransformerProjection,[Awq=>AwqLoRA,init_awq,Nf4=>Nf4LoRA,init_nf4]);
adapt_original_projection!(AdaptedProjection,[]);
adapt_original_projection!(QuantizedTransformerProjection,[Quantized=>QuantizedLoRA,init_quantized]);

impl<B:Backend> From<Linear<B>> for UniversalTransformerProjection<B> {
    fn from(layer:Linear<B>) -> Self {Self::Mixed(MixedTransformerProjection::Dense(layer))}
}
impl<B:Backend> AdaptTransformerProjection<B> for UniversalTransformerProjection<B> {
    fn validate_adapter(&self,config:&TransformerAdapterConfig) {
        match self {Self::Mixed(layer)=>layer.validate_adapter(config),Self::Generic(layer)=>layer.validate_adapter(config)}
    }
    fn with_adapter(self,config:&TransformerAdapterConfig) -> Self {
        self.validate_adapter(config);
        match self {Self::Mixed(layer)=>Self::Mixed(layer.with_adapter(config)),Self::Generic(layer)=>Self::Generic(layer.with_adapter(config))}
    }
}

impl<B:Backend,P:TransformerProjectionShape<B>+From<Linear<B>>> AwqGroupedQueryAttention<B,P> {
    /// Consume actual original dense projections, preserving IDs/trainability and attention geometry.
    /// This does not quantize a projection; packed roles must be supplied as actual packed modules.
    pub fn from_dense(attention:GroupedQueryAttention<B>) -> Self {
        Self::from_projections(attention.query.into(),attention.key.into(),attention.value.into(),attention.output.into(),
            attention.query_heads,attention.kv_heads,attention.head_dimension,attention.dropout)
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>> AwqGroupedQueryAttention<B,P> {
    pub(super) fn validate_adapters(&self,config:&TransformerAdapterConfig,targets:&[AttentionAdapterTarget]) {
        unique(targets);
        for target in targets {
            match target {AttentionAdapterTarget::Query=>self.query.validate_adapter(config),AttentionAdapterTarget::Key=>self.key.validate_adapter(config),
                AttentionAdapterTarget::Value=>self.value.validate_adapter(config),AttentionAdapterTarget::Output=>self.output.validate_adapter(config)}
        }
    }
    /// Add only the explicitly selected attention A/B leaves; untouched roles retain their flags.
    pub fn with_adapters(mut self,config:&TransformerAdapterConfig,targets:&[AttentionAdapterTarget]) -> Self {
        self.validate_adapters(config,targets);
        if targets.contains(&AttentionAdapterTarget::Query) {self.query=self.query.with_adapter(config);}
        if targets.contains(&AttentionAdapterTarget::Key) {self.key=self.key.with_adapter(config);}
        if targets.contains(&AttentionAdapterTarget::Value) {self.value=self.value.with_adapter(config);}
        if targets.contains(&AttentionAdapterTarget::Output) {self.output=self.output.with_adapter(config);}
        self
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>+From<Linear<B>>> AwqFeedForward<B,P> {
    /// Consume actual original dense/gated FFN leaves without allocation or altered activation.
    pub fn from_dense(feed:DenseFeedForward<B>) -> Self {
        Self::from_projections(feed.up.into(),feed.gate.map(Into::into),feed.down.into(),feed.activation,feed.dropout)
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>> AwqFeedForward<B,P> {
    pub(super) fn validate_adapters(&self,config:&TransformerAdapterConfig,targets:&[FeedForwardAdapterTarget]) {
        unique(targets);
        for target in targets {
            match target {FeedForwardAdapterTarget::Up=>self.up.validate_adapter(config),FeedForwardAdapterTarget::Down=>self.down.validate_adapter(config),
                FeedForwardAdapterTarget::Gate=>self.gate.as_ref().expect("selected original FFN has no gate").validate_adapter(config)}
        }
    }
    /// Add only explicitly selected ordinary/gated FFN A/B leaves; no missing gate is created.
    pub fn with_adapters(mut self,config:&TransformerAdapterConfig,targets:&[FeedForwardAdapterTarget]) -> Self {
        self.validate_adapters(config,targets);
        if targets.contains(&FeedForwardAdapterTarget::Up) {self.up=self.up.with_adapter(config);}
        if targets.contains(&FeedForwardAdapterTarget::Gate) {self.gate=self.gate.map(|gate|gate.with_adapter(config));}
        if targets.contains(&FeedForwardAdapterTarget::Down) {self.down=self.down.with_adapter(config);}
        self
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>+From<Linear<B>>> AwqTransformerBlock<B,P> {
    /// Connect an actual original dense block to the shared graph without quantizing its values.
    pub fn from_dense(block:DenseTransformerBlock<B>) -> Self {
        Self {attention:AwqGroupedQueryAttention::from_dense(block.attention),feed_forward:AwqFeedForward::from_dense(block.feed_forward),
            attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first}
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>> AwqTransformerBlock<B,P> {
    pub(super) fn validate_adapters(&self,config:&TransformerAdapterConfig,attention:&[AttentionAdapterTarget],feed_forward:&[FeedForwardAdapterTarget]) {
        assert!(!attention.is_empty() || !feed_forward.is_empty(),"selected native block requires an actual adapter target");
        self.attention.validate_adapters(config,attention);self.feed_forward.validate_adapters(config,feed_forward);
    }
    /// Validate all selected self-attention/FFN roles before allocating any adapter in this block.
    pub fn with_adapters(mut self,config:&TransformerAdapterConfig,attention:&[AttentionAdapterTarget],feed_forward:&[FeedForwardAdapterTarget]) -> Self {
        self.validate_adapters(config,attention,feed_forward);
        self.attention=self.attention.with_adapters(config,attention);self.feed_forward=self.feed_forward.with_adapters(config,feed_forward);self
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>+From<Linear<B>>> AwqTransformerStack<B,P> {
    /// Connect actual original loaded dense blocks in their original order, with no new values.
    pub fn from_dense(stack:DenseTransformerStack<B>) -> Self {Self {blocks:stack.blocks.into_iter().map(AwqTransformerBlock::from_dense).collect()}}
}
impl<B:Backend,P:AdaptTransformerProjection<B>> AwqTransformerStack<B,P> {
    pub(super) fn selected_adapters<'a>(&self,targets:&'a [LayerAdapterConfig]) -> BTreeMap<usize,&'a LayerAdapterConfig> {
        let mut selected=BTreeMap::new();
        for target in targets {
            assert!(target.layer<self.blocks.len(),"native adapter layer index is outside the original stack");
            assert!(selected.insert(target.layer,target).is_none(),"duplicate native adapter layer index");
            self.blocks[target.layer].validate_adapters(&target.adapter,&target.attention,&target.feed_forward);
        }
        selected
    }
    /// Select actual zero-based layers/roles with independent rank/dtype/rsLoRA settings.
    /// All targets are checked first; every unselected base/module identity remains unchanged.
    pub fn with_adapters(self,targets:&[LayerAdapterConfig]) -> Self {
        let selected=self.selected_adapters(targets);
        let blocks=self.blocks.into_iter().enumerate().map(|(index,block)| {
            if let Some(target)=selected.get(&index) {block.with_adapters(&target.adapter,&target.attention,&target.feed_forward)} else {block}
        }).collect();Self {blocks}
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>+From<Linear<B>>> AwqTransformerModel<B,P> {
    /// Connect original dense components without changing their actual values or parameter IDs.
    /// Explicit caller-loaded packed replacements can subsequently occupy independent graph roles.
    pub fn from_dense_parts(embeddings:TransformerEmbeddings<B>,backbone:DenseTransformerStack<B>,normalization:Option<DenseTransformerNorm<B>>,
        head:TransformerHead<B>) -> Self {
        let head=super::AwqTransformerHead::from_projection(head.projection.into(),head.normalization,head.dropout);
        Self::from_parts(embeddings,AwqTransformerStack::from_dense(backbone),normalization,head)
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>> AwqTransformerModel<B,P> {
    /// Add only explicitly selected backbone/head adapters, preserving original input tables and norms.
    /// Packed selections require explicit A/B dtype; there is no guessed dtype from NF4 scales.
    pub fn with_adapters(mut self,targets:&[LayerAdapterConfig],head:Option<&TransformerAdapterConfig>) -> Self {
        let _=self.backbone.selected_adapters(targets);
        if let Some(config)=head {self.head.projection.validate_adapter(config);}
        self.backbone=self.backbone.with_adapters(targets);
        if let Some(config)=head {self.head.projection=self.head.projection.with_adapter(config);}
        self
    }
}

impl<B:Backend,P:TransformerProjectionShape<B>+From<Linear<B>>> ProjectedCrossAttentionBlock<B,P> {
    /// Consume actual original cross-attention leaves without guessing source/target dimensions.
    pub fn from_dense(cross:DenseCrossAttentionBlock<B>) -> Self {
        Self::from_parts(AwqGroupedQueryAttention::from_dense(cross.attention),cross.query_norm,cross.memory_norm,cross.residual_dropout,cross.norm_first)
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>> ProjectedCrossAttentionBlock<B,P> {
    /// Attach adapters only to the explicitly selected real query/memory/output roles.
    pub fn with_adapters(mut self,config:&TransformerAdapterConfig,targets:&[AttentionAdapterTarget]) -> Self {
        self.attention=self.attention.with_adapters(config,targets);self
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>+From<Linear<B>>> ProjectedEncoderDecoderLayer<B,P> {
    /// Preserve actual original self-attention, memory attention and FFN leaves and identities.
    pub fn from_dense(layer:DenseEncoderDecoderLayer<B>) -> Self {
        Self::from_parts(AwqTransformerBlock::from_dense(layer.backbone),ProjectedCrossAttentionBlock::from_dense(layer.cross_attention))
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>> ProjectedEncoderDecoderLayer<B,P> {
    fn validate_adapters(&self,config:&DecoderLayerAdapterConfig) {
        assert!(config.backbone.is_some() || config.cross_attention.is_some(),"selected paired layer requires an actual adapter stage");
        if let Some(config)=&config.backbone {self.backbone.validate_adapters(&config.adapter,&config.attention,&config.feed_forward);}
        if let Some(config)=&config.cross_attention {
            assert!(!config.attention.is_empty(),"selected cross stage requires an actual projection target");
            self.cross_attention.attention.validate_adapters(&config.adapter,&config.attention);
        }
    }
    /// Independently select self/FFN and memory-attention adapter roles without altering stage order.
    pub fn with_adapters(mut self,config:&DecoderLayerAdapterConfig) -> Self {
        self.validate_adapters(config);
        if let Some(config)=&config.backbone {self.backbone=self.backbone.with_adapters(&config.adapter,&config.attention,&config.feed_forward);}
        if let Some(config)=&config.cross_attention {self.cross_attention=self.cross_attention.with_adapters(&config.adapter,&config.attention);}
        self
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>+From<Linear<B>>> ProjectedEncoderDecoderStack<B,P> {
    /// Connect every original dense paired layer, in original order, with no new projection values.
    pub fn from_dense(stack:DenseEncoderDecoderStack<B>) -> Self {Self {layers:stack.layers.into_iter().map(ProjectedEncoderDecoderLayer::from_dense).collect()}}
}
impl<B:Backend,P:AdaptTransformerProjection<B>> ProjectedEncoderDecoderStack<B,P> {
    fn selected_adapters<'a>(&self,targets:&'a [DecoderLayerAdapterConfig]) -> BTreeMap<usize,&'a DecoderLayerAdapterConfig> {
        let mut selected=BTreeMap::new();
        for target in targets {
            assert!(target.layer<self.layers.len(),"paired adapter layer is outside the original decoder stack");
            assert!(selected.insert(target.layer,target).is_none(),"duplicate paired adapter layer index");self.layers[target.layer].validate_adapters(target);
        }
        selected
    }
    /// Validate all original decoder selections first, then allocate only explicitly selected A/B leaves.
    pub fn with_adapters(self,targets:&[DecoderLayerAdapterConfig]) -> Self {
        let selected=self.selected_adapters(targets);
        let layers=self.layers.into_iter().enumerate().map(|(index,layer)| {
            if let Some(config)=selected.get(&index) {layer.with_adapters(config)} else {layer}
        }).collect();Self {layers}
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>+From<Linear<B>>> ProjectedEncoderDecoderModel<B,P> {
    /// Consume original actual independent source/target dense components without initialization or quantization.
    pub fn from_dense_parts(source_embeddings:TransformerEmbeddings<B>,encoder:DenseTransformerStack<B>,encoder_normalization:Option<DenseTransformerNorm<B>>,
        target_embeddings:TransformerEmbeddings<B>,decoder:DenseEncoderDecoderStack<B>,decoder_normalization:Option<DenseTransformerNorm<B>>,head:TransformerHead<B>) -> Self {
        let head=super::AwqTransformerHead::from_projection(head.projection.into(),head.normalization,head.dropout);
        Self::from_parts(source_embeddings,AwqTransformerStack::from_dense(encoder),encoder_normalization,target_embeddings,
            ProjectedEncoderDecoderStack::from_dense(decoder),decoder_normalization,head)
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>> ProjectedEncoderDecoderModel<B,P> {
    /// Apply independent source, target-self/FFN, target-memory and output-head adapter selections.
    /// All selections are checked before any allocation; untouched original values and flags remain intact.
    pub fn with_adapters(mut self,encoder:&[LayerAdapterConfig],decoder:&[DecoderLayerAdapterConfig],head:Option<&TransformerAdapterConfig>) -> Self {
        let _=self.encoder.selected_adapters(encoder);let _=self.decoder.selected_adapters(decoder);
        if let Some(config)=head {self.head.projection.validate_adapter(config);}
        self.encoder=self.encoder.with_adapters(encoder);self.decoder=self.decoder.with_adapters(decoder);
        if let Some(config)=head {self.head.projection=self.head.projection.with_adapter(config);}
        self
    }
}
