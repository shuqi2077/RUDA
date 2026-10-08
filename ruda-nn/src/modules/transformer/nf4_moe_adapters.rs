use alloc::{collections::BTreeMap,vec::Vec};
use ruda_model::tensor::backend::Backend;
use crate::{Nf4MoeLayer,FrozenExpertGeometry};
use super::{AdaptTransformerProjection,TransformerAdapterConfig,AttentionAdapterTarget,FeedForwardAdapterTarget,
    Nf4MoeTransformerBlock,Nf4MoeTransformerLayer,Nf4MoeTransformerModel};

/// Explicit projection roles for actual loaded dense, floating-expert or packed-expert architecture.
#[derive(Clone,Debug)]
pub enum Nf4MoeAdapterTargets {
    /// Original ordinary/gated FFN roles, not expert cube slices.
    Dense(Vec<FeedForwardAdapterTarget>),
    /// Actual router/shared roles of an original floating expert layer.
    Floating {
        /// Adapt the actual original router only when explicitly selected.
        router:bool,
        /// Actual optional source shared FFN projection roles.
        shared:Vec<FeedForwardAdapterTarget>,
    },
    /// Actual router/shared roles of an original packed expert layer.
    Packed {
        /// Adapt the actual original router only when explicitly selected.
        router:bool,
        /// Actual optional source shared FFN projection roles.
        shared:Vec<FeedForwardAdapterTarget>,
    },
}
/// Exact loaded zero-based layer and original independent adapter policy.
#[derive(Clone,Debug)]
pub struct Nf4MoeLayerAdapterConfig {
    /// Actual layer index in the original loaded model.
    pub layer:usize,
    /// Explicit native rank/alpha/dropout/dtype/rsLoRA selection.
    pub adapter:TransformerAdapterConfig,
    /// Actual original attention projection roles.
    pub attention:Vec<AttentionAdapterTarget>,
    /// Actual source branch and selected native projection roles.
    pub feed_forward:Nf4MoeAdapterTargets,
}
impl<B:Backend,P:AdaptTransformerProjection<B>,E:FrozenExpertGeometry<B>> Nf4MoeLayer<B,P,E> {
    /// Attach actual A/B to the explicitly selected router without reblocking or changing expert bytes.
    pub fn with_router_adapter(mut self,config:&TransformerAdapterConfig) -> Self {
        self.router.validate_adapter(config);self.router=self.router.with_adapter(config);self
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>,E:FrozenExpertGeometry<B>> Nf4MoeTransformerBlock<B,P,E> {
    fn validate_adapters(&self,config:&TransformerAdapterConfig,attention:&[AttentionAdapterTarget],router:bool,shared:&[FeedForwardAdapterTarget]) {
        assert!(!attention.is_empty() || router || !shared.is_empty(),"selected packed expert block requires an actual projection target");
        self.attention.validate_adapters(config,attention);if router {self.routed.router.validate_adapter(config);}
        if !shared.is_empty() {self.shared.as_ref().expect("selected packed expert block has no shared branch").validate_adapters(config,shared);}
    }
    /// Attach only selected native attention/router/shared adapters, leaving every quantized expert parameter intact.
    pub fn with_adapters(mut self,config:&TransformerAdapterConfig,attention:&[AttentionAdapterTarget],router:bool,shared:&[FeedForwardAdapterTarget]) -> Self {
        self.validate_adapters(config,attention,router,shared);self.attention=self.attention.with_adapters(config,attention);
        if router {self.routed=self.routed.with_router_adapter(config);}
        if !shared.is_empty() {self.shared=self.shared.map(|branch|branch.with_adapters(config,shared));}self
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>,E:FrozenExpertGeometry<B>> Nf4MoeTransformerLayer<B,P,E> {
    fn validate_adapters(&self,config:&Nf4MoeLayerAdapterConfig) {
        match (self,&config.feed_forward) {
            (Self::Dense(block),Nf4MoeAdapterTargets::Dense(targets))=>block.validate_adapters(&config.adapter,&config.attention,targets),
            (Self::Floating(block),Nf4MoeAdapterTargets::Floating {router,shared})=>block.validate_adapters(&config.adapter,&config.attention,*router,shared),
            (Self::Packed(block),Nf4MoeAdapterTargets::Packed {router,shared})=>block.validate_adapters(&config.adapter,&config.attention,*router,shared),
            _=>panic!("adapter branch target differs from the actual original dense/floating/packed expert layer"),
        }
    }
    /// Consume the actual original layer variant; branch mismatches fail before adapter allocation.
    pub fn with_adapters(self,config:&Nf4MoeLayerAdapterConfig) -> Self {
        self.validate_adapters(config);match (self,&config.feed_forward) {
            (Self::Dense(block),Nf4MoeAdapterTargets::Dense(targets))=>Self::Dense(block.with_adapters(&config.adapter,&config.attention,targets)),
            (Self::Floating(block),Nf4MoeAdapterTargets::Floating {router,shared})=>Self::Floating(block.with_adapters(&config.adapter,&config.attention,*router,shared)),
            (Self::Packed(block),Nf4MoeAdapterTargets::Packed {router,shared})=>Self::Packed(block.with_adapters(&config.adapter,&config.attention,*router,shared)),
            _=>unreachable!("validated actual original layer variant"),
        }
    }
}
impl<B:Backend,P:AdaptTransformerProjection<B>,E:FrozenExpertGeometry<B>> Nf4MoeTransformerModel<B,P,E> {
    /// Validate every target before allocation, then attach only selected original native projection adapters.
    /// Original packed bytes, FP32 scales/book and all unselected parameter identities remain unchanged.
    pub fn with_adapters(mut self,targets:&[Nf4MoeLayerAdapterConfig],head:Option<&TransformerAdapterConfig>) -> Self {
        let mut selected=BTreeMap::new();
        for config in targets {assert!(config.layer<self.layers.len(),"NF4 MoE adapter index exceeds original loaded layers");
            assert!(selected.insert(config.layer,config).is_none(),"duplicate NF4 MoE adapter layer index");self.layers[config.layer].validate_adapters(config);}
        if let Some(config)=head {self.head.projection.validate_adapter(config);}
        self.layers=self.layers.into_iter().enumerate().map(|(index,layer)|if let Some(config)=selected.get(&index) {layer.with_adapters(config)}else {layer}).collect();
        if let Some(config)=head {self.head.projection=self.head.projection.with_adapter(config);}self
    }
}
