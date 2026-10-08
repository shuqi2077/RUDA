use super::{FrozenExpertGeometry,FrozenSelectedExperts,FrozenNf4SwiGluExperts,FrozenPackedSwiGluExperts,
    AdaptedPackedSwiGluExperts,AdaptedFloatingSwiGluExperts,AdaptedExpertError,FloatingExpertError,Nf4MoeLayer};
use ruda_model::{module::{Module,ParamId},tensor::{Tensor,Int,FrozenPackedExpertOps,ExpertProjectionOps,NativeSwiGluOps,backend::Backend}};
use alloc::vec::Vec;
use core::fmt;

/// Original expert payloads eligible for real packed-base A/B attachment, without numerical format conversion.
pub trait PackedExpertAdapterSource<B:Backend>:FrozenExpertGeometry<B> {
    /// Move the actual original NF4/AWQ payloads into independently selectable gate/up/down roles.
    fn into_packed_expert_source(self) -> FrozenPackedSwiGluExperts<B>;
}
impl<B:Backend> PackedExpertAdapterSource<B> for FrozenPackedSwiGluExperts<B> {
    fn into_packed_expert_source(self) -> FrozenPackedSwiGluExperts<B> {self}
}
impl<B:Backend> PackedExpertAdapterSource<B> for FrozenNf4SwiGluExperts<B> {
    fn into_packed_expert_source(self) -> FrozenPackedSwiGluExperts<B> {
        FrozenPackedSwiGluExperts::from_parts(self.gate.into(),self.up.into(),self.down.into())
    }
}

/// Actual selected expert execution in a mixed floating/NF4/AWQ model, retaining the original unselected implementation.
#[derive(Module,Debug)]
pub enum MixedAdaptedExperts<B:Backend,E:Module<B> =FrozenPackedSwiGluExperts<B>> {
    /// Original whole-expert implementation, including its original native VJP and execution policies.
    Original(E),
    /// Actual selected A/B on original immutable NF4/AWQ payloads.
    Packed(AdaptedPackedSwiGluExperts<B>),
    /// Actual selected A/B on original floating cubes; unselected roles retain their original training flags.
    Floating(AdaptedFloatingSwiGluExperts<B>),
}
impl<B:Backend,E:Module<B>> MixedAdaptedExperts<B,E> {
    /// Canonical actual selected A/B parameter identities; the original whole-expert implementation adds none.
    pub fn adapter_parameter_ids(&self) -> Vec<ParamId> {match self {
        Self::Original(_)=>Vec::new(),Self::Packed(value)=>value.adapter_parameter_ids(),Self::Floating(value)=>value.adapter_parameter_ids()}}
}
impl<B:Backend,E:FrozenExpertGeometry<B>> FrozenExpertGeometry<B> for MixedAdaptedExperts<B,E> {
    fn dimensions(&self) -> [usize;3] {match self {
        Self::Original(value)=>value.dimensions(),Self::Packed(value)=>value.dimensions(),Self::Floating(value)=>value.dimensions()}}
    fn validate(&self) {match self {
        Self::Original(value)=>value.validate(),Self::Packed(value)=>value.validate(),Self::Floating(value)=>value.validate()}}
    fn device(&self) -> B::Device {match self {
        Self::Original(value)=>value.device(),Self::Packed(value)=>value.device(),Self::Floating(value)=>value.device()}}
}
/// Source-specific failures from the actual selected original or adapted expert implementation.
#[derive(Debug)]
pub enum MixedExpertError<O:fmt::Debug,P:fmt::Debug,G:fmt::Debug,S:fmt::Debug> {
    /// Original unmodified expert-chain failure.
    Original(O),
    /// Actual packed base, native A/B or original native activation failure.
    Packed(AdaptedExpertError<P,G,S>),
    /// Actual floating base, native A/B or original native activation failure.
    Floating(FloatingExpertError<G,S>),
}
impl<O:fmt::Debug,P:fmt::Debug,G:fmt::Debug,S:fmt::Debug> fmt::Display for MixedExpertError<O,P,G,S> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {
        Self::Original(error)=>write!(f,"original expert chain: {error:?}"),Self::Packed(error)=>write!(f,"{error}"),Self::Floating(error)=>write!(f,"{error}")}}
}
impl<O:fmt::Debug,P:fmt::Debug,G:fmt::Debug,S:fmt::Debug> core::error::Error for MixedExpertError<O,P,G,S> {}
impl<B:FrozenPackedExpertOps+ExpertProjectionOps+NativeSwiGluOps,E:FrozenSelectedExperts<B>> FrozenSelectedExperts<B> for MixedAdaptedExperts<B,E> {
    type Error=MixedExpertError<E::Error,B::PackedExpertError,B::ExpertProjectionError,B::SwiGluError>;
    fn forward(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,expert_start:usize) -> Result<Tensor<B,2>,Self::Error> {
        match self {
            Self::Original(value)=>value.forward(input,ids,expert_start).map_err(MixedExpertError::Original),
            Self::Packed(value)=>value.forward(input,ids,expert_start).map_err(MixedExpertError::Packed),
            Self::Floating(value)=>value.forward(input,ids,expert_start).map_err(MixedExpertError::Floating),
        }
    }
}

/// Complete native route/combine layer containing an actual selected original or adapted expert implementation.
pub type MixedAdaptedMoeLayer<B,P,E=FrozenPackedSwiGluExperts<B>> = Nf4MoeLayer<B,P,MixedAdaptedExperts<B,E>>;
/// Complete native token-to-logit graph with explicit floating/NF4/AWQ expert adaptation and unchanged unselected chains.
pub type MixedAdaptedMoeTransformerModel<B,P,E=FrozenPackedSwiGluExperts<B>> = super::transformer::Nf4MoeTransformerModel<B,P,MixedAdaptedExperts<B,E>>;
/// Actual original native attention/shared/residual block with mixed selected expert execution.
pub type MixedAdaptedMoeTransformerBlock<B,P,E=FrozenPackedSwiGluExperts<B>> = super::transformer::Nf4MoeTransformerBlock<B,P,MixedAdaptedExperts<B,E>>;
/// Explicit original ordinary/native floating/mixed-adapted layer order.
pub type MixedAdaptedMoeTransformerLayer<B,P,E=FrozenPackedSwiGluExperts<B>> = super::transformer::Nf4MoeTransformerLayer<B,P,MixedAdaptedExperts<B,E>>;
