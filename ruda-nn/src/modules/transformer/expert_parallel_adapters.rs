use alloc::{collections::{BTreeMap,BTreeSet},vec::Vec};
use ruda_model::{module::ParamId,tensor::backend::Backend};
use crate::{OwnedFloatingExpertAdapters,SelectableOwnedExperts,AdaptedExpertParallelTransformerModel,
    expert_parallel::{ExpertParallelMoeLayer,ExpertParallelSwiGluExperts}};
use super::{TransformerProjectionShape,FloatingExpertLayerAdapterConfig,ExpertParallelTransformerModel,ExpertParallelTransformerLayer,ExpertParallelTransformerBlock};

/// Explicit zero-based expert-owned layer and original local A/B initialization choices.
pub type ExpertParallelLayerAdapterConfig = FloatingExpertLayerAdapterConfig;
impl<B:Backend,P:TransformerProjectionShape<B>> ExpertParallelTransformerModel<B,P,ExpertParallelSwiGluExperts<B>> {
    /// Attach actual rank-owned expert A/B after validating all source layer/role choices before allocation.
    /// Every unselected layer retains its complete original native execution; no replicated full expert model is created.
    pub fn with_expert_adapters(self,targets:&[ExpertParallelLayerAdapterConfig]) -> AdaptedExpertParallelTransformerModel<B,P> {
        let mut selected=BTreeMap::new();
        for config in targets {
            assert!(config.layer<self.layers.len(),"owned expert adapter layer index exceeds original loaded model");
            assert!(selected.insert(config.layer,config).is_none(),"duplicate original owned expert adapter layer index");
            let ExpertParallelTransformerLayer::Parallel(block)=&self.layers[config.layer] else {
                panic!("selected expert adapters require an actual rank-owned original layer")};
            block.routed.validate();OwnedFloatingExpertAdapters::from_native(block.routed.experts.clone(),block.routed.options.forward,block.routed.options.backward)
                .experts.validate_adapter_targets(&config.adapter,&config.targets,config.adapter_dtype);
        }
        let layers=self.layers.into_iter().enumerate().map(|(index,layer)|match layer {
            ExpertParallelTransformerLayer::Local(layer)=>ExpertParallelTransformerLayer::Local(layer),
            ExpertParallelTransformerLayer::Parallel(block)=>{
                let routed=block.routed;let experts=if let Some(config)=selected.get(&index) {
                    SelectableOwnedExperts::Adapted(OwnedFloatingExpertAdapters::from_native(routed.experts,routed.options.forward,routed.options.backward)
                        .with_adapters(&config.adapter,&config.targets,config.adapter_dtype,config.use_rslora,config.forward,config.backward))
                } else {SelectableOwnedExperts::Original(routed.experts)};
                ExpertParallelTransformerLayer::Parallel(ExpertParallelTransformerBlock {attention:block.attention,
                    routed:ExpertParallelMoeLayer::from_expert_parts(routed.router,experts,routed.correction_bias,routed.options,routed.router_input_dtype),
                    shared:block.shared,attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first})
            },
        }).collect();
        ExpertParallelTransformerModel::from_expert_parts(self.embeddings,layers,self.normalization,self.head)
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>> AdaptedExpertParallelTransformerModel<B,P> {
    /// Canonical actual locally owned A/B IDs only; no expert-world replicated gradient reduction is inferred.
    pub fn expert_adapter_parameter_ids(&self) -> Vec<ParamId> {
        let mut ids=BTreeSet::new();for layer in &self.layers {if let ExpertParallelTransformerLayer::Parallel(block)=layer {
            ids.extend(block.routed.experts.adapter_parameter_ids());}}ids.into_iter().collect()
    }
    /// Merge only explicitly present local expert A/B and return original native rank-owned model execution.
    /// Original expert-world ownership, router/shared/head/attention parameters, cache topology and training flags remain intact.
    pub fn merge_expert_adapters(self) -> ExpertParallelTransformerModel<B,P> {
        for layer in &self.layers {if let ExpertParallelTransformerLayer::Parallel(block)=layer {
            if let SelectableOwnedExperts::Adapted(value)=&block.routed.experts {value.experts.base_strategies();}}}
        let layers=self.layers.into_iter().map(|layer|match layer {
            ExpertParallelTransformerLayer::Local(layer)=>ExpertParallelTransformerLayer::Local(layer),
            ExpertParallelTransformerLayer::Parallel(block)=>{
                let routed=block.routed;let (experts,options)=match routed.experts {
                    SelectableOwnedExperts::Original(value)=>(value,routed.options),
                    SelectableOwnedExperts::Adapted(value)=>{
                        let (forward,backward)=value.experts.base_strategies();let mut options=routed.options;options.forward=forward;options.backward=backward;
                        (value.merge(),options)
                    },
                };
                ExpertParallelTransformerLayer::Parallel(ExpertParallelTransformerBlock {attention:block.attention,
                    routed:ExpertParallelMoeLayer::from_parts(routed.router,experts,routed.correction_bias,options,routed.router_input_dtype),
                    shared:block.shared,attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first})
            },
        }).collect();
        ExpertParallelTransformerModel::from_parts(self.embeddings,layers,self.normalization,self.head)
    }
}
