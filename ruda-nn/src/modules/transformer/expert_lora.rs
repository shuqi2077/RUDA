use alloc::{collections::BTreeMap,vec::Vec};
use ruda_model::{module::ParamId,tensor::{DType,MoeExpertStrategy,backend::Backend}};
use crate::{LoRALinearConfig,ExpertAdapterTarget,FrozenPackedSwiGluExperts,AdaptedPackedSwiGluExperts,SelectablePackedExperts,
    SelectablePackedMoeTransformerModel,Nf4MoeLayer,AdaptedFloatingSwiGluExperts,AdaptedFloatingMoeTransformerModel,
    PackedExpertAdapterSource,MixedAdaptedExperts,MixedAdaptedMoeTransformerModel,FrozenExpertGeometry};
use super::{TransformerProjectionShape,Nf4MoeTransformerModel,Nf4MoeTransformerLayer,Nf4MoeTransformerBlock,
    NativeMoeTransformerModel,NativeMoeTransformerStack,NativeMoeTransformerLayer,NativeMoeTransformerBlock,NativeMoeFeedForward};

/// Actual loaded zero-based expert layer and explicit original expert adapter choices.
#[derive(Clone,Debug)]
pub struct PackedExpertLayerAdapterConfig {
    /// Actual original layer index, not a model-family inferred name.
    pub layer:usize,
    /// Original explicit rank/alpha/dropout choices.
    pub adapter:LoRALinearConfig,
    /// Actual gate/up/down roles; unselected roles retain original parameters and trainability.
    pub targets:Vec<ExpertAdapterTarget>,
    /// Explicit native FP32/FP16/BF16 adapter leaf storage.
    pub adapter_dtype:DType,
    /// Select original alpha/sqrt(rank), rather than alpha/rank, only when explicitly true.
    pub use_rslora:bool,
    /// Original actual floating adapter grouped forward policy.
    pub forward:MoeExpertStrategy,
    /// Original actual independent floating adapter grouped backward policy.
    pub backward:MoeExpertStrategy,
}
/// Floating expert layer selection using the same explicit A/B configuration as packed expert layers.
pub type FloatingExpertLayerAdapterConfig = PackedExpertLayerAdapterConfig;
/// Explicit layer/role choices shared by mixed original floating/NF4/AWQ expert models.
pub type ExpertLayerAdapterConfig = PackedExpertLayerAdapterConfig;
impl<B:Backend,P:TransformerProjectionShape<B>> Nf4MoeTransformerModel<B,P,FrozenPackedSwiGluExperts<B>> {
    /// Validate all actual layer/role configurations before allocation, then attach only selected expert A/B.
    /// Original dense/floating layers, unselected packed execution, source payloads and all original IDs remain intact.
    pub fn with_expert_adapters(self,targets:&[PackedExpertLayerAdapterConfig]) -> SelectablePackedMoeTransformerModel<B,P> {
        let mut selected=BTreeMap::new();
        for config in targets {
            assert!(config.layer<self.layers.len(),"expert adapter layer index exceeds original loaded model");
            assert!(selected.insert(config.layer,config).is_none(),"duplicate original expert adapter layer index");
            let Nf4MoeTransformerLayer::Packed(block)=&self.layers[config.layer] else {panic!("selected expert adapters require an actual original packed-expert layer")};
            AdaptedPackedSwiGluExperts::from_frozen(block.routed.experts.clone()).validate_adapter_targets(&config.adapter,&config.targets,config.adapter_dtype);
        }
        let layers=self.layers.into_iter().enumerate().map(|(index,layer)|match layer {
            Nf4MoeTransformerLayer::Dense(block)=>Nf4MoeTransformerLayer::Dense(block),
            Nf4MoeTransformerLayer::Floating(block)=>Nf4MoeTransformerLayer::Floating(block),
            Nf4MoeTransformerLayer::Packed(block)=>{
                let routed=block.routed;
                let experts=if let Some(config)=selected.get(&index) {SelectablePackedExperts::Adapted(AdaptedPackedSwiGluExperts::from_frozen(routed.experts)
                    .with_adapters(&config.adapter,&config.targets,config.adapter_dtype,config.use_rslora,config.forward,config.backward))}
                    else {SelectablePackedExperts::Original(routed.experts)};
                Nf4MoeTransformerLayer::Packed(Nf4MoeTransformerBlock {attention:block.attention,
                    routed:Nf4MoeLayer {router:routed.router,experts,correction_bias:routed.correction_bias,routing:routed.routing,router_input_dtype:routed.router_input_dtype},
                    shared:block.shared,attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first})
            },
        }).collect();
        Nf4MoeTransformerModel::from_parts(self.embeddings,layers,self.normalization,self.head)
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>> Nf4MoeTransformerModel<B,P,SelectablePackedExperts<B>> {
    /// Canonical actual expert A/B IDs only; attention/router/shared/head adapters remain independently selectable.
    pub fn expert_adapter_parameter_ids(&self) -> Vec<ParamId> {
        let mut ids=Vec::new();for layer in &self.layers {if let Nf4MoeTransformerLayer::Packed(block)=layer {
            for id in block.routed.experts.adapter_parameter_ids() {if !ids.contains(&id) {ids.push(id);}}}}ids
    }
}

impl<B:Backend,P:TransformerProjectionShape<B>> NativeMoeTransformerModel<B,P> {
    /// Attach actual expert A/B only to selected loaded native layers, validating every role before any allocation.
    /// Unselected routed layers keep their original whole-expert execution; all other model components remain intact.
    pub fn with_expert_adapters(self,targets:&[FloatingExpertLayerAdapterConfig]) -> AdaptedFloatingMoeTransformerModel<B,P> {
        let mut selected=BTreeMap::new();
        for config in targets {
            assert!(config.layer<self.backbone.layers.len(),"floating expert adapter layer index exceeds original loaded model");
            assert!(selected.insert(config.layer,config).is_none(),"duplicate original floating expert adapter layer index");
            let NativeMoeTransformerLayer::Routed(block)=&self.backbone.layers[config.layer] else {
                panic!("selected floating expert adapters require an actual original routed layer")};
            let routed=&block.feed_forward.routed;routed.validate();
            AdaptedFloatingSwiGluExperts::from_native(routed.experts.clone(),routed.options.forward,routed.options.backward)
                .validate_adapter_targets(&config.adapter,&config.targets,config.adapter_dtype);
        }
        let layers=self.backbone.layers.into_iter().enumerate().map(|(index,layer)|match layer {
            NativeMoeTransformerLayer::Dense(block)=>Nf4MoeTransformerLayer::Dense(block),
            NativeMoeTransformerLayer::Routed(block)=>{
                if let Some(config)=selected.get(&index) {
                    let routed=block.feed_forward.routed.with_expert_adapters(&config.adapter,&config.targets,config.adapter_dtype,
                        config.use_rslora,config.forward,config.backward);
                    Nf4MoeTransformerLayer::Packed(Nf4MoeTransformerBlock {attention:block.attention,routed,shared:block.feed_forward.shared,
                        attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first})
                } else {Nf4MoeTransformerLayer::Floating(block)}
            },
        }).collect();
        Nf4MoeTransformerModel::from_parts(self.embeddings,layers,self.normalization,self.head)
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>> AdaptedFloatingMoeTransformerModel<B,P> {
    /// Canonical actual expert A/B IDs only; original floating cubes and independent projection adapters are excluded.
    pub fn expert_adapter_parameter_ids(&self) -> Vec<ParamId> {
        let mut ids=Vec::new();for layer in &self.layers {if let Nf4MoeTransformerLayer::Packed(block)=layer {
            for id in block.routed.experts.adapter_parameter_ids() {if !ids.contains(&id) {ids.push(id);}}}}ids
    }
    /// Merge only explicitly present expert A/B and restore the complete original native model structure for inference.
    /// Retains embeddings, attention/router/shared/head adapters, original layer order, norms and residual policies.
    pub fn merge_expert_adapters(self) -> NativeMoeTransformerModel<B,P> {
        for layer in &self.layers {if let Nf4MoeTransformerLayer::Packed(block)=layer {
            block.validate();block.routed.experts.base_strategies();}}
        let layers=self.layers.into_iter().map(|layer|match layer {
            Nf4MoeTransformerLayer::Dense(block)=>NativeMoeTransformerLayer::Dense(block),
            Nf4MoeTransformerLayer::Floating(block)=>NativeMoeTransformerLayer::Routed(block),
            Nf4MoeTransformerLayer::Packed(block)=>NativeMoeTransformerLayer::Routed(NativeMoeTransformerBlock {
                attention:block.attention,feed_forward:NativeMoeFeedForward {routed:block.routed.merge_expert_adapters(),shared:block.shared},
                attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first}),
        }).collect();
        NativeMoeTransformerModel::from_parts(self.embeddings,NativeMoeTransformerStack {layers},self.normalization,self.head)
    }
}

impl<B:Backend,P:TransformerProjectionShape<B>,E:PackedExpertAdapterSource<B>> Nf4MoeTransformerModel<B,P,E> {
    /// Adapt actual floating and packed expert layers in one original mixed stack, without guessing model-family targets.
    /// All selections are validated before A/B allocation; every unselected layer retains its original whole-expert path.
    pub fn with_mixed_expert_adapters(self,targets:&[ExpertLayerAdapterConfig]) -> MixedAdaptedMoeTransformerModel<B,P,E> {
        let mut selected=BTreeMap::new();
        for config in targets {
            assert!(config.layer<self.layers.len(),"mixed expert adapter layer index exceeds original loaded model");
            assert!(selected.insert(config.layer,config).is_none(),"duplicate original mixed expert adapter layer index");
            match &self.layers[config.layer] {
                Nf4MoeTransformerLayer::Dense(_)=>panic!("selected mixed expert adapters require an actual routed layer"),
                Nf4MoeTransformerLayer::Floating(block)=>{
                    let routed=&block.feed_forward.routed;routed.validate();
                    AdaptedFloatingSwiGluExperts::from_native(routed.experts.clone(),routed.options.forward,routed.options.backward)
                        .validate_adapter_targets(&config.adapter,&config.targets,config.adapter_dtype);
                },
                Nf4MoeTransformerLayer::Packed(block)=>{
                    block.routed.validate();AdaptedPackedSwiGluExperts::from_frozen(block.routed.experts.clone().into_packed_expert_source())
                        .validate_adapter_targets(&config.adapter,&config.targets,config.adapter_dtype);
                },
            }
        }
        let layers=self.layers.into_iter().enumerate().map(|(index,layer)|match layer {
            Nf4MoeTransformerLayer::Dense(block)=>Nf4MoeTransformerLayer::Dense(block),
            Nf4MoeTransformerLayer::Floating(block)=>{
                if let Some(config)=selected.get(&index) {
                    let routed=block.feed_forward.routed.with_expert_adapters(&config.adapter,&config.targets,config.adapter_dtype,
                        config.use_rslora,config.forward,config.backward);
                    Nf4MoeTransformerLayer::Packed(Nf4MoeTransformerBlock {attention:block.attention,routed:Nf4MoeLayer {
                        router:routed.router,experts:MixedAdaptedExperts::Floating(routed.experts),correction_bias:routed.correction_bias,
                        routing:routed.routing,router_input_dtype:routed.router_input_dtype},shared:block.feed_forward.shared,
                        attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first})
                } else {Nf4MoeTransformerLayer::Floating(block)}
            },
            Nf4MoeTransformerLayer::Packed(block)=>{
                let routed=block.routed;let experts=if let Some(config)=selected.get(&index) {
                    MixedAdaptedExperts::Packed(AdaptedPackedSwiGluExperts::from_frozen(routed.experts.into_packed_expert_source())
                        .with_adapters(&config.adapter,&config.targets,config.adapter_dtype,config.use_rslora,config.forward,config.backward))
                } else {MixedAdaptedExperts::Original(routed.experts)};
                Nf4MoeTransformerLayer::Packed(Nf4MoeTransformerBlock {attention:block.attention,routed:Nf4MoeLayer {
                    router:routed.router,experts,correction_bias:routed.correction_bias,routing:routed.routing,router_input_dtype:routed.router_input_dtype},
                    shared:block.shared,attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first})
            },
        }).collect();
        Nf4MoeTransformerModel::from_parts(self.embeddings,layers,self.normalization,self.head)
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>,E:FrozenExpertGeometry<B>> MixedAdaptedMoeTransformerModel<B,P,E> {
    /// Canonical actual selected expert A/B identities across floating/NF4/AWQ layers.
    pub fn expert_adapter_parameter_ids(&self) -> Vec<ParamId> {
        let mut ids=Vec::new();for layer in &self.layers {if let Nf4MoeTransformerLayer::Packed(block)=layer {
            for id in block.routed.experts.adapter_parameter_ids() {if !ids.contains(&id) {ids.push(id);}}}}ids
    }
}
