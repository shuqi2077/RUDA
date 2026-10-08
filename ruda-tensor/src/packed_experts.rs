//! Actual original AWQ/NF4 selected expert payloads and their native first-order chains.
use crate::{Backend,grouped_nf4::Nf4ExpertPayload,tensor::{FloatTensor,IntTensor}};
use core::fmt;

/// Explicit resident AWQ geometry and global expert range, without guessed source metadata.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct AwqExpertOptions {
    /// Actual expert matrices represented by original rank-three words/scales.
    pub experts:usize,
    /// First global expert ID represented by this resident cube.
    pub expert_start:usize,
    /// Original per-expert input width.
    pub input_features:usize,
    /// Original per-expert output width, divisible by eight.
    pub output_features:usize,
    /// Original complete input-channel group size.
    pub group_size:usize,
}
/// Original AWQ expert cube, with scale storage independent of floating activation storage.
#[derive(Clone,Debug)]
pub struct AwqExpertPayload<B:Backend> {
    /// Original permuted I32 words `[E,K,N/8]`.
    pub qweight:IntTensor<B>,
    /// Original I32 zero-point words `[E,K/group,N/8]`.
    pub qzeros:IntTensor<B>,
    /// Original FP32/FP16/BF16 scales `[E,K/group,N]`.
    pub scales:FloatTensor<B>,
    /// Actual optional original frozen scale-dtype bias `[E,N]`.
    pub bias:Option<FloatTensor<B>>,
    /// Exact source geometry and explicit original global expert range.
    pub options:AwqExpertOptions,
}
/// Source-selected original packed expert format, not a quantizer or dense surrogate.
#[derive(Clone,Debug)]
pub enum PackedExpertPayload<B:Backend> {
    /// Original RUDA high-nibble-first flat-block NF4.
    Nf4(Nf4ExpertPayload<B>),
    /// Original AWQ per-input-group permuted eight-code I32 words.
    Awq(AwqExpertPayload<B>),
}
impl<B:Backend> PackedExpertPayload<B> {
    /// Actual resident expert count and explicitly declared first global expert ID.
    pub fn expert_range(&self) -> (usize,usize) {match self {Self::Nf4(value)=>(value.options.experts,value.options.expert_start),
        Self::Awq(value)=>(value.options.experts,value.options.expert_start)}}
}
/// Original native failure or unsupported differentiation of frozen packed metadata/VJPs.
#[derive(Debug)]
pub enum PackedExpertAutodiffError<E:fmt::Debug> {
    /// Original native projection/routing error.
    Native(E),
    /// Packed scales/codebook/bias must remain frozen; QAT is not this operation.
    TrainableMetadata,
    /// Original native input VJPs provide first-order derivatives only.
    HigherDerivativeUnsupported,
}
impl<E:fmt::Debug> fmt::Display for PackedExpertAutodiffError<E> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {Self::Native(error)=>write!(f,"native packed expert: {error:?}"),
        Self::TrainableMetadata=>f.write_str("packed expert quantization metadata and bias must be frozen"),
        Self::HigherDerivativeUnsupported=>f.write_str("native packed expert input VJPs provide first-order derivatives only")}}
}
impl<E:fmt::Debug> core::error::Error for PackedExpertAutodiffError<E> {}
/// Native grouped projections and SwiGLU with independently selected original gate/up/down formats.
pub trait FrozenPackedExpertOps:Backend {
    /// Original native or first-order derivative contract failure.
    type PackedExpertError:fmt::Debug;
    /// Actual packed operand and native private row mapping.
    type PackedProjectionState:Clone+Send+fmt::Debug+'static;
    /// Actual original row mapping and optional real input VJP intermediates.
    type PackedSwiGluState:Clone+Send+fmt::Debug+'static;
    /// Evaluate only actual assigned experts and restore exact incoming row order.
    fn packed_expert_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,payload:PackedExpertPayload<Self>)
        -> Result<(FloatTensor<Self>,Self::PackedProjectionState),Self::PackedExpertError>;
    /// First-order packed projection input VJP in original activation storage.
    fn packed_expert_input_backward(state:Self::PackedProjectionState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::PackedExpertError>;
    /// Complete native selected gate/up/down and original storage-rounded SwiGLU.
    /// AD preserves required actual input caches even when `retain_input` is false.
    fn packed_swiglu_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,gate:PackedExpertPayload<Self>,up:PackedExpertPayload<Self>,down:PackedExpertPayload<Self>,retain_input:bool)
        -> Result<(FloatTensor<Self>,Self::PackedSwiGluState),Self::PackedExpertError>;
    /// Original first-order input VJP through the actual selected packed chain.
    fn packed_swiglu_input_backward(state:Self::PackedSwiGluState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::PackedExpertError>;
}
