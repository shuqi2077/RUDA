use alloc::{collections::BTreeMap,vec::Vec};
use ruda_model::{module::ParamId,tensor::{DType,MoeExpertStrategy,backend::Backend}};
use crate::{LoRALinearConfig,ExpertAdapterTarget,FrozenPackedSwiGluExperts,AdaptedPackedSwiGluExperts,SelectablePackedExperts,
    SelectablePackedMoeTransformerModel,Nf4MoeLayer};
use super::{TransformerProjectionShape,Nf4MoeTransformerModel,Nf4MoeTransformerLayer,Nf4MoeTransformerBlock};

/// Actual loaded zero-based packed-expert layer and explicit original expert adapter choices.
#[derive(Clone,Debug)]
pub struct PackedExpertLayerAdapterConfig {
    /// Actual original layer index, not a model-family inferred name.
    pub layer:usize,
    /// Original explicit rank/alpha/dropout choices.
    pub adapter:LoRALinearConfig,
    /// Actual gate/up/down roles; unselected roles retain original packed parameters.
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
