use alloc::vec::Vec;
use core::fmt;
use ruda_model::{module::{Module,ParamId},record::RecorderError,tensor::{Tensor,Int,DType,MoeOptions,MoeExpertStrategy,
    MoeReceivedOps,FrozenPackedExpertOps,ExpertProjectionOps,NativeSwiGluOps,backend::Backend}};
use super::{NativeSwiGluExperts,AdaptedFloatingSwiGluExperts,SelectableOwnedExperts,OwnedFloatingExpertAdapters,
    SelectablePackedExperts,OwnedPackedExperts,FrozenPackedSwiGluExperts,FrozenNf4SwiGluExperts,AdaptedPackedSwiGluExperts,
    MixedAdaptedExperts,LoRALinearConfig,ExpertAdapterTarget,FrozenExpertGeometry,FrozenSelectedExperts,
    ExpertAdapterProjections,ExpertAdapterProjectionRef,ExpertAdapterMapper,AdaptedExpertError,FloatingExpertError};
use super::expert_parallel::{ExpertOwnership,ExpertParallelGeometry,ExpertParallelReceived,ExpertParallelMoeLayer};

/// Actual complete loaded source representation, moved into owned execution without numerical conversion.
#[derive(Debug)]
pub enum MixedExpertParallelSource<B:Backend> {
    /// Original whole native floating expert chain, retaining its source layer's policies.
    Native(NativeSwiGluExperts<B>),
    /// Actual original floating projections and any already loaded per-expert A/B.
    Floating(AdaptedFloatingSwiGluExperts<B>),
    /// Actual independently selected original NF4/AWQ projections and any already loaded A/B.
    Packed(SelectablePackedExperts<B>),
}
impl<B:Backend> From<NativeSwiGluExperts<B>> for MixedExpertParallelSource<B> {fn from(value:NativeSwiGluExperts<B>) -> Self {Self::Native(value)}}
impl<B:Backend> From<AdaptedFloatingSwiGluExperts<B>> for MixedExpertParallelSource<B> {fn from(value:AdaptedFloatingSwiGluExperts<B>) -> Self {Self::Floating(value)}}
impl<B:Backend> From<SelectablePackedExperts<B>> for MixedExpertParallelSource<B> {fn from(value:SelectablePackedExperts<B>) -> Self {Self::Packed(value)}}
macro_rules! packed_source {
    ($source:ident)=>{impl<B:Backend> From<$source<B>> for MixedExpertParallelSource<B> {fn from(value:$source<B>) -> Self {Self::Packed(value.into())}}};
}
packed_source!(FrozenPackedSwiGluExperts);
packed_source!(FrozenNf4SwiGluExperts);
packed_source!(AdaptedPackedSwiGluExperts);
impl<B:Backend,E:FrozenExpertGeometry<B>+Into<SelectablePackedExperts<B>>> From<MixedAdaptedExperts<B,E>> for MixedExpertParallelSource<B> {
    fn from(value:MixedAdaptedExperts<B,E>) -> Self {match value {
        MixedAdaptedExperts::Original(value)=>Self::Packed(value.into()),MixedAdaptedExperts::Packed(value)=>Self::Packed(value.into()),
        MixedAdaptedExperts::Floating(value)=>Self::Floating(value)}}
}
impl<B:Backend> MixedExpertParallelSource<B> {
    /// Actual original `[experts,input,intermediate]` widths, without decoding packed values.
    pub fn dimensions(&self) -> [usize;3] {match self {Self::Native(value)=>value.dimensions(),Self::Floating(value)=>value.dimensions(),Self::Packed(value)=>value.dimensions()}}
    /// Validate actual source representation before selecting or allocating owned storage.
    pub fn validate(&self) {match self {Self::Native(value)=>value.validate(),Self::Floating(value)=>value.validate(),Self::Packed(value)=>value.validate()}}
    pub(super) fn validate_adapter_targets(&self,config:&LoRALinearConfig,targets:&[ExpertAdapterTarget],dtype:DType,options:MoeOptions) {
        match self {
            Self::Native(value)=>AdaptedFloatingSwiGluExperts::from_native(value.clone(),options.forward,options.backward).validate_adapter_targets(config,targets,dtype),
            Self::Floating(value)=>value.validate_adapter_targets(config,targets,dtype),
            Self::Packed(SelectablePackedExperts::Original(value))=>AdaptedPackedSwiGluExperts::from_frozen(value.clone()).validate_adapter_targets(config,targets,dtype),
            Self::Packed(SelectablePackedExperts::Adapted(value))=>value.validate_adapter_targets(config,targets,dtype),
        }
    }
}
/// Actual rank-owned expert chain selected independently for every original loaded layer.
#[derive(Module,Debug)]
pub enum MixedOwnedExperts<B:Backend> {
    /// Original whole floating execution or explicitly adapted native floating projections.
    Floating(SelectableOwnedExperts<B>),
    /// Original independent NF4/AWQ source windows and explicitly present native A/B.
    Packed(OwnedPackedExperts<B>),
}
impl<B:Backend> MixedOwnedExperts<B> {
    /// Actual `[owned_experts,hidden,intermediate]` widths.
    pub fn dimensions(&self) -> [usize;3] {match self {Self::Floating(value)=>ExpertParallelGeometry::dimensions(value),Self::Packed(value)=>value.dimensions()}}
    /// Original resident source device.
    pub fn device(&self) -> B::Device {match self {Self::Floating(value)=>ExpertParallelGeometry::device(value),Self::Packed(value)=>value.device()}}
    /// Validate actual resident values against each layer's original explicit ownership.
    pub fn validate(&self) {match self {Self::Floating(value)=>ExpertParallelGeometry::validate(value),Self::Packed(value)=>value.validate()}}
    /// Attach new A/B only to explicit unadapted roles, retaining source base policies independently of adapter policies.
    pub fn with_adapters(self,config:&LoRALinearConfig,targets:&[ExpertAdapterTarget],dtype:DType,use_rslora:bool,
        source_options:MoeOptions,forward:MoeExpertStrategy,backward:MoeExpertStrategy) -> Self {
        let value=match self {
            Self::Floating(SelectableOwnedExperts::Original(value))=>Self::Floating(SelectableOwnedExperts::Adapted(
                OwnedFloatingExpertAdapters::from_native(value,source_options.forward,source_options.backward).with_adapters(config,targets,dtype,use_rslora,forward,backward))),
            Self::Floating(SelectableOwnedExperts::Adapted(value))=>Self::Floating(SelectableOwnedExperts::Adapted(value.with_adapters(config,targets,dtype,use_rslora,forward,backward))),
            Self::Packed(value)=>Self::Packed(value.with_adapters(config,targets,dtype,use_rslora,forward,backward)),
        };value.validate();value
    }
    /// Canonical actual locally owned A/B identities only.
    pub fn adapter_parameter_ids(&self) -> Vec<ParamId> {match self {Self::Floating(value)=>value.adapter_parameter_ids(),Self::Packed(value)=>value.adapter_parameter_ids()}}
}
impl<B:Backend> ExpertParallelGeometry<B> for MixedOwnedExperts<B> {
    fn dimensions(&self) -> [usize;3] {self.dimensions()}
    fn ownership(&self) -> &ExpertOwnership {match self {Self::Floating(value)=>value.ownership(),Self::Packed(value)=>&value.ownership}}
    fn rank(&self) -> usize {match self {Self::Floating(value)=>value.rank(),Self::Packed(value)=>value.rank}}
    fn device(&self) -> B::Device {self.device()}
    fn validate(&self) {self.validate();}
    fn parameter_ids(&self) -> Vec<ParamId> {match self {Self::Floating(value)=>value.parameter_ids(),Self::Packed(value)=>value.parameter_ids()}}
}
/// Original routing or actual selected floating/packed expert failure, preserving native source error types.
#[derive(Debug)]
pub enum MixedOwnedExpertError<M:fmt::Debug,P:fmt::Debug,G:fmt::Debug,S:fmt::Debug> {
    /// Original native routing or unadapted whole floating expert failure.
    Native(M),
    /// Actual source packed projection, A/B or native activation failure.
    Packed(AdaptedExpertError<P,G,S>),
    /// Actual original floating projection, A/B or native activation failure.
    Floating(FloatingExpertError<G,S>),
}
impl<M:fmt::Debug,P:fmt::Debug,G:fmt::Debug,S:fmt::Debug> fmt::Display for MixedOwnedExpertError<M,P,G,S> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {Self::Native(error)=>write!(f,"native owned expert: {error:?}"),Self::Packed(error)=>write!(f,"{error}"),Self::Floating(error)=>write!(f,"{error}")}}
}
impl<M:fmt::Debug,P:fmt::Debug,G:fmt::Debug,S:fmt::Debug> core::error::Error for MixedOwnedExpertError<M,P,G,S> {}
impl<B:MoeReceivedOps+FrozenPackedExpertOps+ExpertProjectionOps+NativeSwiGluOps> ExpertParallelReceived<B> for MixedOwnedExperts<B> {
    type Error=MixedOwnedExpertError<B::MoeError,B::PackedExpertError,B::ExpertProjectionError,B::SwiGluError>;
    fn routing_error(error:B::MoeError) -> Self::Error {MixedOwnedExpertError::Native(error)}
    fn forward_received(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,options:MoeOptions) -> Result<Tensor<B,2>,Self::Error> {
        self.validate();match self {
            Self::Floating(SelectableOwnedExperts::Original(value))=>value.forward_received(input,ids,options).map_err(MixedOwnedExpertError::Native),
            Self::Floating(SelectableOwnedExperts::Adapted(value))=>value.experts.forward(input,ids,value.ownership.range(value.rank).start).map_err(MixedOwnedExpertError::Floating),
            Self::Packed(value)=>value.experts.forward(input,ids,value.ownership.range(value.rank).start).map_err(MixedOwnedExpertError::Packed),
        }
    }
}
impl<B:Backend> FrozenExpertGeometry<B> for MixedOwnedExperts<B> {
    fn dimensions(&self) -> [usize;3] {self.dimensions()}
    fn device(&self) -> B::Device {self.device()}
    fn validate(&self) {self.validate();}
}
impl<B:Backend> ExpertAdapterProjections<B> for MixedOwnedExperts<B> {
    fn expert_adapter_projections(&self) -> Vec<(ExpertAdapterTarget,ExpertAdapterProjectionRef<'_,B>)> {match self {
        Self::Floating(value)=>value.expert_adapter_projections(),Self::Packed(value)=>value.expert_adapter_projections()}}
    fn map_expert_adapters<M:ExpertAdapterMapper<B>>(self,mapper:&mut M) -> Result<Self,RecorderError> {
        let owned=match self {Self::Floating(value)=>Self::Floating(value.map_expert_adapters(mapper)?),Self::Packed(value)=>Self::Packed(value.map_expert_adapters(mapper)?)};
        owned.validate();Ok(owned)
    }
}
/// Complete original expert transport/routing with independently selected floating/NF4/AWQ layers.
pub type MixedExpertParallelMoeLayer<B,P> = ExpertParallelMoeLayer<B,P,MixedOwnedExperts<B>>;
/// Complete original dense/packed/cache/causal-loss execution with actual mixed owned expert layers.
pub type MixedExpertParallelTransformerModel<B,P> = crate::transformer::ExpertParallelTransformerModel<B,P,MixedOwnedExperts<B>>;
