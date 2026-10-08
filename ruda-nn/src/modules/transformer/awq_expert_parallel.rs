use alloc::{collections::{BTreeMap,BTreeSet},vec::Vec};
use ruda_model::{module::ParamId,tensor::backend::Backend};
use crate::{AwqExpertPartitionContext,OwnedAwqExperts,AwqExpertParallelTransformerModel,FrozenPackedSwiGluExperts,
    FrozenPackedExpertProjection,SelectablePackedExperts,AdaptedPackedSwiGluExperts,FrozenExpertGeometry,
    expert_parallel::{ExpertOwnership,ExpertParallelMoeLayer}};
use super::{TransformerProjectionShape,Nf4MoeTransformerModel,Nf4MoeTransformerLayer,NativeMoeTransformerLayer,
    ExpertParallelTransformerModel,ExpertParallelTransformerLayer,ExpertParallelTransformerBlock,PackedExpertLayerAdapterConfig};

/// Actual loaded AWQ expert layer and explicitly declared expert-world ownership.
#[derive(Clone,Debug)]
pub struct AwqExpertParallelLayerConfig {
    /// Actual zero-based source layer index.
    pub layer:usize,
    /// Original explicit complete expert-world intervals, including empty owners.
    pub ownership:ExpertOwnership,
    /// Optional new local A/B initialization; absence preserves original packed expert execution.
    pub adapters:Option<PackedExpertLayerAdapterConfig>,
}
impl<B:Backend,P:TransformerProjectionShape<B>> Nf4MoeTransformerModel<B,P,FrozenPackedSwiGluExperts<B>> {
    /// Copy only explicit rank-owned AWQ payloads, retaining the original complete model and native expert transport graph.
    /// Every packed layer needs an explicit ownership entry; original dense/floating layers remain local and unchanged.
    pub fn into_awq_expert_parallel(self,targets:&[AwqExpertParallelLayerConfig],rank:usize) -> AwqExpertParallelTransformerModel<B,P> {
        let mut selected=BTreeMap::new();
        for config in targets {
            assert!(config.layer<self.layers.len(),"AWQ expert-owned layer index exceeds original loaded model");
            assert!(selected.insert(config.layer,config).is_none(),"duplicate original AWQ expert-owned layer index");
            let Nf4MoeTransformerLayer::Packed(block)=&self.layers[config.layer] else {
                panic!("AWQ ownership requires an actual original packed expert layer")};
            block.validate();assert_eq!(block.routed.experts.dimensions()[0],config.ownership.experts(),"actual original AWQ count differs from declared expert ownership");
            config.ownership.range(rank);
            for projection in [&block.routed.experts.gate,&block.routed.experts.up,&block.routed.experts.down] {
                assert!(matches!(projection,FrozenPackedExpertProjection::Awq(_)),"AWQ expert ownership requires original AWQ storage; no packed format conversion is performed");}
            if let Some(adapters)=&config.adapters {
                assert_eq!(adapters.layer,config.layer,"AWQ expert adapter and ownership layer indices differ");
                AdaptedPackedSwiGluExperts::from_frozen(block.routed.experts.clone())
                    .validate_adapter_targets(&adapters.adapter,&adapters.targets,adapters.adapter_dtype);
            }
        }
        for (index,layer) in self.layers.iter().enumerate() {if matches!(layer,Nf4MoeTransformerLayer::Packed(_)) {
            assert!(selected.contains_key(&index),"every original packed expert layer requires explicit AWQ ownership before distributed conversion");}}
        let mut context=AwqExpertPartitionContext::new();
        let layers=self.layers.into_iter().enumerate().map(|(index,layer)|match layer {
            Nf4MoeTransformerLayer::Dense(block)=>ExpertParallelTransformerLayer::Local(NativeMoeTransformerLayer::Dense(block)),
            Nf4MoeTransformerLayer::Floating(block)=>ExpertParallelTransformerLayer::Local(NativeMoeTransformerLayer::Routed(block)),
            Nf4MoeTransformerLayer::Packed(block)=>{
                let config=selected[&index];let routed=block.routed;
                let mut experts=OwnedAwqExperts::from_full(SelectablePackedExperts::Original(routed.experts),config.ownership.clone(),rank,&mut context);
                if let Some(adapters)=&config.adapters {experts=experts.with_adapters(&adapters.adapter,&adapters.targets,adapters.adapter_dtype,
                    adapters.use_rslora,adapters.forward,adapters.backward);}
                let options=ruda_model::tensor::MoeOptions {selection:routed.routing.selection,weights:routed.routing.weights,
                    combine_backward:routed.routing.combine_backward,forward:ruda_model::tensor::MoeExpertStrategy::Scalar,backward:ruda_model::tensor::MoeExpertStrategy::Scalar};
                ExpertParallelTransformerLayer::Parallel(ExpertParallelTransformerBlock {attention:block.attention,
                    routed:ExpertParallelMoeLayer::from_expert_parts(routed.router,experts,routed.correction_bias,options,routed.router_input_dtype),
                    shared:block.shared,attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first})
            },
        }).collect();
        ExpertParallelTransformerModel::from_expert_parts(self.embeddings,layers,self.normalization,self.head)
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>> AwqExpertParallelTransformerModel<B,P> {
    /// Canonical actual owned A/B identities only; quantization metadata and model projection adapters remain independent.
    pub fn expert_adapter_parameter_ids(&self) -> Vec<ParamId> {
        let mut ids=BTreeSet::new();for layer in &self.layers {if let ExpertParallelTransformerLayer::Parallel(block)=layer {
            ids.extend(block.routed.experts.adapter_parameter_ids());}}ids.into_iter().collect()
    }
}
