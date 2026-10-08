use alloc::{collections::{BTreeMap,BTreeSet},vec::Vec};
use ruda_model::{module::ParamId,tensor::{backend::Backend,MoeOptions,MoeExpertStrategy}};
use crate::{PackedExpertPartitionContext,OwnedPackedExperts,PackedExpertParallelTransformerModel,SelectablePackedExperts,
    AdaptedPackedSwiGluExperts,FrozenExpertGeometry,expert_parallel::{ExpertOwnership,ExpertParallelMoeLayer}};
use super::{TransformerProjectionShape,Nf4MoeTransformerModel,Nf4MoeTransformerLayer,NativeMoeTransformerLayer,
    ExpertParallelTransformerModel,ExpertParallelTransformerLayer,ExpertParallelTransformerBlock,PackedExpertLayerAdapterConfig};

/// Actual loaded packed expert layer and complete original explicit expert-world ownership.
#[derive(Clone,Debug)]
pub struct PackedExpertParallelLayerConfig {
    /// Actual zero-based source layer index.
    pub layer:usize,
    /// Original explicit expert-world intervals, including empty owners.
    pub ownership:ExpertOwnership,
    /// Optional new local A/B for explicitly unadapted roles; existing loaded A/B are preserved.
    pub adapters:Option<PackedExpertLayerAdapterConfig>,
}
impl<B:Backend,P:TransformerProjectionShape<B>,E:FrozenExpertGeometry<B>+Into<SelectablePackedExperts<B>>> Nf4MoeTransformerModel<B,P,E> {
    /// Copy only rank-owned original NF4/AWQ payloads and actual loaded A/B into the complete native model graph.
    /// NF4 retains original intersecting scale blocks and the first coefficient's in-block offset.
    /// Every packed layer requires explicit ownership; original dense/floating layers remain local and unchanged.
    pub fn into_packed_expert_parallel(self,targets:&[PackedExpertParallelLayerConfig],rank:usize) -> PackedExpertParallelTransformerModel<B,P> {
        let mut selected=BTreeMap::new();
        for config in targets {
            assert!(config.layer<self.layers.len(),"packed expert-owned layer index exceeds actual loaded model");
            assert!(selected.insert(config.layer,config).is_none(),"duplicate actual packed expert-owned layer index");
            let Nf4MoeTransformerLayer::Packed(block)=&self.layers[config.layer] else {panic!("packed ownership requires an actual loaded packed expert layer")};
            block.validate();assert_eq!(block.routed.experts.dimensions()[0],config.ownership.experts(),"actual packed expert count differs from declared ownership");
            config.ownership.range(rank);
            if let Some(adapters)=&config.adapters {
                assert_eq!(adapters.layer,config.layer,"packed expert adapter and ownership layer indices differ");
                let source:SelectablePackedExperts<B>=block.routed.experts.clone().into();
                let adapted=match source {SelectablePackedExperts::Original(value)=>AdaptedPackedSwiGluExperts::from_frozen(value),SelectablePackedExperts::Adapted(value)=>value};
                adapted.validate_adapter_targets(&adapters.adapter,&adapters.targets,adapters.adapter_dtype);
            }
        }
        for (index,layer) in self.layers.iter().enumerate() {if matches!(layer,Nf4MoeTransformerLayer::Packed(_)) {
            assert!(selected.contains_key(&index),"every actual packed expert layer needs explicit ownership before distributed conversion");}}
        let mut context=PackedExpertPartitionContext::new();
        let layers=self.layers.into_iter().enumerate().map(|(index,layer)|match layer {
            Nf4MoeTransformerLayer::Dense(block)=>ExpertParallelTransformerLayer::Local(NativeMoeTransformerLayer::Dense(block)),
            Nf4MoeTransformerLayer::Floating(block)=>ExpertParallelTransformerLayer::Local(NativeMoeTransformerLayer::Routed(block)),
            Nf4MoeTransformerLayer::Packed(block)=>{
                let config=selected[&index];let routed=block.routed;
                let mut experts=OwnedPackedExperts::from_full(routed.experts.into(),config.ownership.clone(),rank,&mut context);
                if let Some(adapters)=&config.adapters {experts=experts.with_adapters(&adapters.adapter,&adapters.targets,adapters.adapter_dtype,
                    adapters.use_rslora,adapters.forward,adapters.backward);}
                let options=MoeOptions {selection:routed.routing.selection,weights:routed.routing.weights,combine_backward:routed.routing.combine_backward,
                    forward:MoeExpertStrategy::Scalar,backward:MoeExpertStrategy::Scalar};
                ExpertParallelTransformerLayer::Parallel(ExpertParallelTransformerBlock {attention:block.attention,
                    routed:ExpertParallelMoeLayer::from_expert_parts(routed.router,experts,routed.correction_bias,options,routed.router_input_dtype),
                    shared:block.shared,attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first})
            },
        }).collect();
        ExpertParallelTransformerModel::from_expert_parts(self.embeddings,layers,self.normalization,self.head)
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>> PackedExpertParallelTransformerModel<B,P> {
    /// Canonical actual owned expert A/B identities, excluding base quantization and unrelated model adapters.
    pub fn expert_adapter_parameter_ids(&self) -> Vec<ParamId> {
        let mut ids=BTreeSet::new();for layer in &self.layers {if let ExpertParallelTransformerLayer::Parallel(block)=layer {
            ids.extend(block.routed.experts.adapter_parameter_ids());}}ids.into_iter().collect()
    }
}
