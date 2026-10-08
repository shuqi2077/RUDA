use alloc::{collections::{BTreeMap,BTreeSet},vec::Vec};
use ruda_model::{module::ParamId,tensor::{backend::Backend,MoeOptions,MoeExpertStrategy}};
use crate::{PackedExpertPartitionContext,MixedExpertParallelSource,MixedOwnedExperts,MixedExpertParallelTransformerModel,
    FrozenExpertGeometry,Nf4MoeRouting,expert_parallel::{ExpertOwnership,ExpertParallelMoeLayer}};
use super::{TransformerProjectionShape,Nf4MoeTransformerModel,Nf4MoeTransformerLayer,NativeMoeTransformerLayer,
    ExpertParallelTransformerModel,ExpertParallelTransformerLayer,ExpertParallelTransformerBlock,PackedExpertLayerAdapterConfig};

/// Actual source layer ownership and optional new expert A/B, independent of floating/NF4/AWQ storage.
#[derive(Clone,Debug)]
pub struct MixedExpertParallelLayerConfig {
    /// Actual zero-based loaded layer index.
    pub layer:usize,
    /// Original explicit complete expert-world intervals.
    pub ownership:ExpertOwnership,
    /// Optional new local A/B for actual unadapted roles; existing loaded adapters remain intact.
    pub adapters:Option<PackedExpertLayerAdapterConfig>,
}
fn packed_options(routing:Nf4MoeRouting) -> MoeOptions {
    MoeOptions {selection:routing.selection,weights:routing.weights,combine_backward:routing.combine_backward,
        forward:MoeExpertStrategy::Scalar,backward:MoeExpertStrategy::Scalar}
}
fn owned<B:Backend>(source:MixedExpertParallelSource<B>,options:MoeOptions,config:&MixedExpertParallelLayerConfig,rank:usize,
    context:&mut PackedExpertPartitionContext<B>) -> MixedOwnedExperts<B> {
    let mut experts=context.mixed_experts(source,config.ownership.clone(),rank);
    if let Some(adapters)=&config.adapters {experts=experts.with_adapters(&adapters.adapter,&adapters.targets,adapters.adapter_dtype,
        adapters.use_rslora,options,adapters.forward,adapters.backward);}experts
}
impl<B:Backend,P:TransformerProjectionShape<B>,E:FrozenExpertGeometry<B>+Into<MixedExpertParallelSource<B>>> Nf4MoeTransformerModel<B,P,E> {
    /// Copy only explicit rank-owned floating/NF4/AWQ source layers and present A/B into one complete original model graph.
    /// Packed layers require ownership entries; unselected original floating and dense layers remain local and unchanged.
    /// Floating base policies and every quantized projection's original execution policy remain independent of adapter policies.
    pub fn into_mixed_expert_parallel(self,targets:&[MixedExpertParallelLayerConfig],rank:usize) -> MixedExpertParallelTransformerModel<B,P> {
        let mut selected=BTreeMap::new();
        for config in targets {
            assert!(config.layer<self.layers.len(),"mixed owned-expert layer index exceeds actual loaded model");
            assert!(selected.insert(config.layer,config).is_none(),"duplicate actual mixed owned-expert layer index");
            let (source,options)=match &self.layers[config.layer] {
                Nf4MoeTransformerLayer::Dense(_)=>panic!("expert ownership cannot select an original non-expert dense layer"),
                Nf4MoeTransformerLayer::Floating(block)=>{block.validate();(MixedExpertParallelSource::Native(block.routed.experts.clone()),block.routed.options)},
                Nf4MoeTransformerLayer::Packed(block)=>{block.validate();(block.routed.experts.clone().into(),packed_options(block.routed.routing))},
            };
            source.validate();assert_eq!(source.dimensions()[0],config.ownership.experts(),"actual source expert count differs from declared mixed ownership");config.ownership.range(rank);
            if let Some(adapters)=&config.adapters {assert_eq!(adapters.layer,config.layer,"mixed adapter and ownership layer indices differ");
                source.validate_adapter_targets(&adapters.adapter,&adapters.targets,adapters.adapter_dtype,options);}
        }
        for (index,layer) in self.layers.iter().enumerate() {if matches!(layer,Nf4MoeTransformerLayer::Packed(_)) {
            assert!(selected.contains_key(&index),"every actual packed expert layer requires explicit ownership before mixed distributed conversion");}}
        let mut context=PackedExpertPartitionContext::new();
        let layers=self.layers.into_iter().enumerate().map(|(index,layer)|match layer {
            Nf4MoeTransformerLayer::Dense(block)=>ExpertParallelTransformerLayer::Local(NativeMoeTransformerLayer::Dense(block)),
            Nf4MoeTransformerLayer::Floating(block)=>{
                let Some(config)=selected.get(&index) else {return ExpertParallelTransformerLayer::Local(NativeMoeTransformerLayer::Routed(block))};
                let routed=block.routed;let options=routed.options;let experts=owned(routed.experts.into(),options,config,rank,&mut context);
                ExpertParallelTransformerLayer::Parallel(ExpertParallelTransformerBlock {attention:block.attention,
                    routed:ExpertParallelMoeLayer::from_expert_parts(routed.router,experts,routed.correction_bias,options,routed.router_input_dtype),
                    shared:block.shared,attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first})
            },
            Nf4MoeTransformerLayer::Packed(block)=>{
                let config=selected[&index];let routed=block.routed;let options=packed_options(routed.routing);let experts=owned(routed.experts.into(),options,config,rank,&mut context);
                ExpertParallelTransformerLayer::Parallel(ExpertParallelTransformerBlock {attention:block.attention,
                    routed:ExpertParallelMoeLayer::from_expert_parts(routed.router,experts,routed.correction_bias,options,routed.router_input_dtype),
                    shared:block.shared,attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first})
            },
        }).collect();
        ExpertParallelTransformerModel::from_expert_parts(self.embeddings,layers,self.normalization,self.head)
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>> MixedExpertParallelTransformerModel<B,P> {
    /// Canonical actual rank-owned A/B identities, without inferring gradient-reduction groups.
    pub fn expert_adapter_parameter_ids(&self) -> Vec<ParamId> {
        let mut ids=BTreeSet::new();for layer in &self.layers {if let ExpertParallelTransformerLayer::Parallel(block)=layer {
            ids.extend(block.routed.experts.adapter_parameter_ids());}}ids.into_iter().collect()
    }
}
