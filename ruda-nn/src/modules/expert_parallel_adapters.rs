use super::expert_parallel::{ExpertOwnership,ExpertPartitionContext,ExpertParallelSwiGluExperts,ExpertParallelGeometry,ExpertParallelReceived,ExpertParallelMoeLayer};
use crate::{ExpertLinear,FloatingExpertProjection,AdaptedFloatingSwiGluExperts,FloatingExpertError,FloatingExpertLoRAError,
    LoRALinearConfig,ExpertAdapterTarget,ExpertAdapterProjections,ExpertAdapterProjectionRef,ExpertAdapterMapper,FrozenExpertGeometry,FrozenSelectedExperts,
    transformer::TransformerProjectionShape};
use alloc::{collections::BTreeSet,vec::Vec};
use ruda_model::{module::{Module,ModuleVisitor,Param,ParamId},tensor::{Tensor,Int,DType,MoeOptions,MoeExpertStrategy,MoeReceivedOps,MoeOps,
    ExpertProjectionOps,NativeSwiGluOps,backend::Backend},record::RecorderError};

/// Actual local floating expert cubes/A/B with the original complete expert-world ownership.
#[derive(Module,Debug)]
pub struct OwnedFloatingExpertAdapters<B:Backend> {
    /// Only actual rank-owned original bases and present A/B, including empty-owner axes.
    pub experts:AdaptedFloatingSwiGluExperts<B>,
    /// Original explicitly declared global expert ownership.
    #[module(skip)]
    pub ownership:ExpertOwnership,
    /// Original expert-world rank, independent of the data/tensor rank.
    pub rank:usize,
}
impl<B:Backend> OwnedFloatingExpertAdapters<B> {
    /// Actual local `[owned_experts,hidden,intermediate]` widths.
    pub fn dimensions(&self) -> [usize;3] {self.experts.dimensions()}
    /// Actual original resident base/adapter device.
    pub fn device(&self) -> B::Device {self.experts.device()}
    /// Validate original native storage and actual explicitly owned expert count.
    pub fn validate(&self) {
        self.experts.validate();assert_eq!(self.dimensions()[0],self.ownership.range(self.rank).len(),"actual adapted expert count differs from owned interval");
    }
    /// Connect actual caller-loaded local cubes/A/B without generating absent expert matrices.
    pub fn from_parts(experts:AdaptedFloatingSwiGluExperts<B>,ownership:ExpertOwnership,rank:usize) -> Self {
        let owned=Self {experts,ownership,rank};owned.validate();owned
    }
    /// Retain original local base values/IDs/flags and source execution before explicit role adaptation.
    pub fn from_native(source:ExpertParallelSwiGluExperts<B>,forward:MoeExpertStrategy,backward:MoeExpertStrategy) -> Self {
        source.validate();let experts=AdaptedFloatingSwiGluExperts::from_parts(
            FloatingExpertProjection::Dense(ExpertLinear::from_parameters(source.gate,forward,backward)),
            FloatingExpertProjection::Dense(ExpertLinear::from_parameters(source.up,forward,backward)),
            FloatingExpertProjection::Dense(ExpertLinear::from_parameters(source.down,forward,backward)));
        Self::from_parts(experts,source.ownership,source.rank)
    }
    /// Copy only this rank's original full-model bases and A/B, reusing canonical original tied leaves.
    /// No values are downloaded, no global expert replica is retained and no rank split is inferred.
    pub fn from_full(source:AdaptedFloatingSwiGluExperts<B>,ownership:ExpertOwnership,rank:usize,context:&mut ExpertPartitionContext<B>) -> Self {
        source.validate();assert_eq!(source.dimensions()[0],ownership.experts(),"actual global adapted expert count differs from original ownership");
        let mut projection=|value|match value {
            FloatingExpertProjection::Dense(mut linear)=>{linear.weight=context.parameter(linear.weight,&ownership,rank);FloatingExpertProjection::Dense(linear)},
            FloatingExpertProjection::LoRA(mut layer)=>{
                layer.base.weight=context.parameter(layer.base.weight,&ownership,rank);
                layer.adapter_a.weight=context.parameter(layer.adapter_a.weight,&ownership,rank);
                layer.adapter_b.weight=context.parameter(layer.adapter_b.weight,&ownership,rank);FloatingExpertProjection::LoRA(layer)
            },
        };
        let experts=AdaptedFloatingSwiGluExperts::from_parts(projection(source.gate),projection(source.up),projection(source.down));
        Self::from_parts(experts,ownership,rank)
    }
    /// Initialize actual A/B only on explicitly selected owned roles, including real empty-owner parameters.
    /// Original base execution is independent of the caller-selected adapter execution/dtype.
    pub fn with_adapters(mut self,config:&LoRALinearConfig,targets:&[ExpertAdapterTarget],dtype:DType,use_rslora:bool,
        forward:MoeExpertStrategy,backward:MoeExpertStrategy) -> Self {
        self.experts=self.experts.with_adapters(config,targets,dtype,use_rslora,forward,backward);self.validate();self
    }
    /// Canonical actual owned A/B identities only, excluding original base cubes.
    pub fn adapter_parameter_ids(&self) -> Vec<ParamId> {self.experts.adapter_parameter_ids()}
    /// Merge selected local A/B for inference, restoring only this rank's original cube-only native execution.
    pub fn merge(self) -> ExpertParallelSwiGluExperts<B> {
        let gate=self.experts.gate.merge().weight;let up=self.experts.up.merge().weight;let down=self.experts.down.merge().weight;
        ExpertParallelSwiGluExperts::from_parameters(gate,up,down,self.ownership,self.rank)
    }
}
struct OwnedIds(BTreeSet<ParamId>);
impl<B:Backend> ModuleVisitor<B> for OwnedIds {
    fn visit_float<const D:usize>(&mut self,parameter:&Param<Tensor<B,D>>) {self.0.insert(parameter.id);}
}
impl<B:Backend> ExpertParallelGeometry<B> for OwnedFloatingExpertAdapters<B> {
    fn dimensions(&self) -> [usize;3] {self.experts.dimensions()}
    fn ownership(&self) -> &ExpertOwnership {&self.ownership}
    fn rank(&self) -> usize {self.rank}
    fn device(&self) -> B::Device {self.experts.device()}
    fn validate(&self) {
        self.validate();
    }
    fn parameter_ids(&self) -> Vec<ParamId> {let mut ids=OwnedIds(BTreeSet::new());self.experts.visit(&mut ids);ids.0.into_iter().collect()}
}
impl<B> ExpertParallelReceived<B> for OwnedFloatingExpertAdapters<B>
where B:MoeReceivedOps+ExpertProjectionOps<ExpertProjectionError=<B as MoeOps>::MoeError>+NativeSwiGluOps<SwiGluError=<B as MoeOps>::MoeError> {
    type Error=B::MoeError;
    fn routing_error(error:B::MoeError) -> Self::Error {error}
    fn forward_received(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,_options:MoeOptions) -> Result<Tensor<B,2>,B::MoeError> {
        self.validate();self.experts.forward(input,ids,self.ownership.range(self.rank).start).map_err(|error|match error {
            FloatingExpertError::Projection(FloatingExpertLoRAError::Base(error)|FloatingExpertLoRAError::Adapter(error))=>error,
            FloatingExpertError::Activation(error)=>error,
        })
    }
}
/// Preserve original whole native received-expert execution on unselected layers.
#[derive(Module,Debug)]
pub enum SelectableOwnedExperts<B:Backend> {
    /// Original rank-owned cube-only native chain and its original VJP.
    Original(ExpertParallelSwiGluExperts<B>),
    /// Actual explicitly selected local base/A/B native chain.
    Adapted(OwnedFloatingExpertAdapters<B>),
}
impl<B:Backend> SelectableOwnedExperts<B> {
    /// Actual local expert A/B IDs only; the unmodified native chain contains none.
    pub fn adapter_parameter_ids(&self) -> Vec<ParamId> {match self {Self::Original(_)=>Vec::new(),Self::Adapted(value)=>value.adapter_parameter_ids()}}
}
impl<B:Backend> ExpertParallelGeometry<B> for SelectableOwnedExperts<B> {
    fn dimensions(&self) -> [usize;3] {match self {Self::Original(value)=>value.dimensions(),Self::Adapted(value)=>value.dimensions()}}
    fn ownership(&self) -> &ExpertOwnership {match self {Self::Original(value)=>&value.ownership,Self::Adapted(value)=>&value.ownership}}
    fn rank(&self) -> usize {match self {Self::Original(value)=>value.rank,Self::Adapted(value)=>value.rank}}
    fn device(&self) -> B::Device {match self {Self::Original(value)=>value.device(),Self::Adapted(value)=>value.device()}}
    fn validate(&self) {match self {Self::Original(value)=>value.validate(),Self::Adapted(value)=>value.validate()}}
    fn parameter_ids(&self) -> Vec<ParamId> {match self {Self::Original(value)=>value.parameter_ids(),Self::Adapted(value)=>value.parameter_ids()}}
}
impl<B> ExpertParallelReceived<B> for SelectableOwnedExperts<B>
where B:MoeReceivedOps+ExpertProjectionOps<ExpertProjectionError=<B as MoeOps>::MoeError>+NativeSwiGluOps<SwiGluError=<B as MoeOps>::MoeError> {
    type Error=B::MoeError;
    fn routing_error(error:B::MoeError) -> Self::Error {error}
    fn forward_received(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,options:MoeOptions) -> Result<Tensor<B,2>,B::MoeError> {
        match self {Self::Original(value)=>value.forward_received(input,ids,options),Self::Adapted(value)=>value.forward_received(input,ids,options)}
    }
}
impl<B:Backend> ExpertAdapterProjections<B> for OwnedFloatingExpertAdapters<B> {
    fn expert_adapter_projections(&self) -> Vec<(ExpertAdapterTarget,ExpertAdapterProjectionRef<'_,B>)> {self.experts.expert_adapter_projections()}
    fn map_expert_adapters<M:ExpertAdapterMapper<B>>(mut self,mapper:&mut M) -> Result<Self,RecorderError> {
        self.experts=self.experts.map_expert_adapters(mapper)?;self.validate();Ok(self)
    }
}
impl<B:Backend> FrozenExpertGeometry<B> for OwnedFloatingExpertAdapters<B> {
    fn dimensions(&self) -> [usize;3] {self.experts.dimensions()}
    fn validate(&self) {<Self as ExpertParallelGeometry<B>>::validate(self);}
    fn device(&self) -> B::Device {self.experts.device()}
}
impl<B:Backend,P:TransformerProjectionShape<B>> ExpertParallelMoeLayer<B,P> {
    /// Attach actual A/B only to explicit local expert roles, preserving the original router/bias/transport configuration.
    pub fn with_expert_adapters(self,config:&LoRALinearConfig,targets:&[ExpertAdapterTarget],dtype:DType,use_rslora:bool,
        forward:MoeExpertStrategy,backward:MoeExpertStrategy) -> ExpertParallelMoeLayer<B,P,OwnedFloatingExpertAdapters<B>> {
        self.validate();let experts=OwnedFloatingExpertAdapters::from_native(self.experts,self.options.forward,self.options.backward)
            .with_adapters(config,targets,dtype,use_rslora,forward,backward);
        ExpertParallelMoeLayer::from_expert_parts(self.router,experts,self.correction_bias,self.options,self.router_input_dtype)
    }
}
/// Complete original expert-owned routed graph with real local A/B and explicit cross-rank transport.
pub type AdaptedExpertParallelMoeLayer<B,P> = ExpertParallelMoeLayer<B,P,SelectableOwnedExperts<B>>;
/// Complete original dense/packed/cache/causal-loss graph with selected rank-owned expert A/B.
pub type AdaptedExpertParallelTransformerModel<B,P> = crate::transformer::ExpertParallelTransformerModel<B,P,SelectableOwnedExperts<B>>;
