use alloc::{collections::BTreeMap,vec::Vec};
use ruda_model::tensor::backend::Backend;
use crate::NativeMoeLayer;
use super::{AdaptTransformerProjection,TransformerAdapterConfig,AttentionAdapterTarget,FeedForwardAdapterTarget,
    NativeMoeFeedForward,NativeMoeTransformerBlock,NativeMoeTransformerLayer,NativeMoeTransformerStack,NativeMoeTransformerModel};

/// Explicit actual feed-forward projection targets, distinguishing dense and routed architecture.
#[derive(Clone,Debug)]
pub enum NativeMoeFeedForwardAdapterTargets {
    /// Original ordinary/gated dense FFN roles, not expert cube slices.
    Dense(Vec<FeedForwardAdapterTarget>),
    /// Original routed router and explicitly present shared FFN roles.
    Routed {
        /// Attach A/B to the actual router projection only when selected.
        router:bool,
        /// Actual shared ordinary/gated projections; empty retains the shared branch unchanged.
        shared:Vec<FeedForwardAdapterTarget>,
    },
}
/// Original zero-based layer and explicit native projection adapter policy.
#[derive(Clone,Debug)]
pub struct NativeMoeLayerAdapterConfig {
    /// Actual loaded layer index.
    pub layer:usize,
    /// Explicit original rank/alpha/dropout/dtype/rsLoRA settings.
    pub adapter:TransformerAdapterConfig,
    /// Actual original attention roles.
    pub attention:Vec<AttentionAdapterTarget>,
    /// Actual original dense or routed/shared branch roles.
    pub feed_forward:NativeMoeFeedForwardAdapterTargets,
}
impl<B:Backend,P:AdaptTransformerProjection<B>> NativeMoeLayer<B,P> {
    /// Add actual native A/B leaves to the router only. Expert cubes/correction bias retain original values and flags.
    pub fn with_router_adapter(mut self,config:&TransformerAdapterConfig) -> Self {
        self.router=self.router.with_adapter(config);self
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>> NativeMoeFeedForward<B,P> {
    fn validate_adapters(&self,config:&TransformerAdapterConfig,router:bool,shared:&[FeedForwardAdapterTarget]) {
        if router {self.routed.router.validate_adapter(config);}
        if !shared.is_empty() {self.shared.as_ref().expect("selected original routed FFN has no shared branch").validate_adapters(config,shared);}
    }
    /// Attach only explicitly selected router/shared native adapters; no expert LoRA arithmetic is invented.
    pub fn with_adapters(mut self,config:&TransformerAdapterConfig,router:bool,shared:&[FeedForwardAdapterTarget]) -> Self {
        self.validate_adapters(config,router,shared);if router {self.routed=self.routed.with_router_adapter(config);}
        if !shared.is_empty() {self.shared=self.shared.map(|branch|branch.with_adapters(config,shared));}self
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>> NativeMoeTransformerBlock<B,P> {
    pub(super) fn validate_adapters(&self,config:&TransformerAdapterConfig,attention:&[AttentionAdapterTarget],router:bool,shared:&[FeedForwardAdapterTarget]) {
        assert!(!attention.is_empty() || router || !shared.is_empty(),"selected routed block requires an actual native projection target");
        self.attention.validate_adapters(config,attention);self.feed_forward.validate_adapters(config,router,shared);
    }
    /// Preserve original attention/expert/shared stage order while attaching selected native adapters.
    pub fn with_adapters(mut self,config:&TransformerAdapterConfig,attention:&[AttentionAdapterTarget],router:bool,shared:&[FeedForwardAdapterTarget]) -> Self {
        self.validate_adapters(config,attention,router,shared);self.attention=self.attention.with_adapters(config,attention);
        self.feed_forward=self.feed_forward.with_adapters(config,router,shared);self
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>> NativeMoeTransformerLayer<B,P> {
    fn validate_adapters(&self,config:&NativeMoeLayerAdapterConfig) {
        match (self,&config.feed_forward) {
            (Self::Dense(block),NativeMoeFeedForwardAdapterTargets::Dense(targets))=>block.validate_adapters(&config.adapter,&config.attention,targets),
            (Self::Routed(block),NativeMoeFeedForwardAdapterTargets::Routed {router,shared})=>block.validate_adapters(&config.adapter,&config.attention,*router,shared),
            _=>panic!("native adapter target branch differs from the actual loaded dense/routed layer"),
        }
    }
    /// Consume only the actual original layer variant; incompatible branch targets are rejected before allocation.
    pub fn with_adapters(self,config:&NativeMoeLayerAdapterConfig) -> Self {
        self.validate_adapters(config);match (self,&config.feed_forward) {
            (Self::Dense(block),NativeMoeFeedForwardAdapterTargets::Dense(targets))=>Self::Dense(block.with_adapters(&config.adapter,&config.attention,targets)),
            (Self::Routed(block),NativeMoeFeedForwardAdapterTargets::Routed {router,shared})=>Self::Routed(block.with_adapters(&config.adapter,&config.attention,*router,shared)),
            _=>unreachable!("validated original native layer variant"),
        }
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>> NativeMoeTransformerStack<B,P> {
    fn selected_adapters<'a>(&self,targets:&'a [NativeMoeLayerAdapterConfig]) -> BTreeMap<usize,&'a NativeMoeLayerAdapterConfig> {
        let mut selected=BTreeMap::new();for config in targets {
            assert!(config.layer<self.layers.len(),"native MoE adapter layer index exceeds actual loaded layers");
            assert!(selected.insert(config.layer,config).is_none(),"duplicate native MoE adapter layer index");self.layers[config.layer].validate_adapters(config);
        }selected
    }
    /// Validate every actual target first, then attach only selected attention/dense/router/shared A/B leaves.
    pub fn with_adapters(self,targets:&[NativeMoeLayerAdapterConfig]) -> Self {
        let selected=self.selected_adapters(targets);Self {layers:self.layers.into_iter().enumerate().map(|(index,layer)| {
            if let Some(config)=selected.get(&index) {layer.with_adapters(config)} else {layer}
        }).collect()}
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>> NativeMoeTransformerModel<B,P> {
    /// Complete native model adapter selection with explicit packed A/B dtype and unchanged unselected storage.
    pub fn with_adapters(mut self,targets:&[NativeMoeLayerAdapterConfig],head:Option<&TransformerAdapterConfig>) -> Self {
        let _=self.backbone.selected_adapters(targets);if let Some(config)=head {self.head.projection.validate_adapter(config);}
        self.backbone=self.backbone.with_adapters(targets);if let Some(config)=head {self.head.projection=self.head.projection.with_adapter(config);}self
    }
}
