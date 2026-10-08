//! Original native floating expert projections and storage-rounded SwiGLU input VJPs.
use crate::{Backend,moe::MoeExpertStrategy,tensor::{FloatTensor,IntTensor}};
use core::fmt;

/// Explicit original grouped execution, independent of quantized base execution choices.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct ExpertProjectionOptions {
    /// First global expert ID represented by actual original `[E,N,K]` weights.
    pub expert_start:usize,
    /// Original actual floating grouped forward policy.
    pub forward:MoeExpertStrategy,
    /// Original actual independent floating grouped backward policy.
    pub backward:MoeExpertStrategy,
}
/// Requested original native input/cube derivatives.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct ExpertProjectionSelection {pub input:bool,pub weights:bool}
/// Actual original optional VJP outputs. Native weight derivatives retain FP32.
#[derive(Debug)]
pub struct ExpertProjectionBackward<B:Backend> {pub input:Option<FloatTensor<B>>,pub weights:Option<FloatTensor<B>>}
/// Original actual grouped projection, including real trainable expert adapter matrices.
pub trait ExpertProjectionOps:Backend {
    /// Original native row/projection or first-order differentiation failure.
    type ExpertProjectionError:fmt::Debug;
    /// Original actual floating cube and native private COPY row mapping.
    type ExpertProjectionState:Clone+Send+fmt::Debug+'static;
    /// Execute actual U32 assigned expert projections and restore original incoming row order.
    fn expert_projection_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,weights:FloatTensor<Self>,options:ExpertProjectionOptions)
        -> Result<(FloatTensor<Self>,Self::ExpertProjectionState),Self::ExpertProjectionError>;
    /// Native first-order selected derivatives, without allocations for omitted cube/input gradients.
    fn expert_projection_backward(state:Self::ExpertProjectionState,gradient:FloatTensor<Self>,selection:ExpertProjectionSelection)
        -> Result<ExpertProjectionBackward<Self>,Self::ExpertProjectionError>;
}
/// Actual requested native storage-rounded activation input derivatives.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct NativeSwiGluSelection {pub gate:bool,pub up:bool}
/// Original actual optional activation input VJPs.
#[derive(Debug)]
pub struct NativeSwiGluBackward<B:Backend> {pub gate:Option<FloatTensor<B>>,pub up:Option<FloatTensor<B>>}
/// Original ruDNN activation arithmetic, not an alternate unfused AD derivative formula.
pub trait NativeSwiGluOps:Backend {
    /// Original native activation or first-order differentiation failure.
    type SwiGluError:fmt::Debug;
    /// Round stored SiLU before multiplication, retaining original activation storage and geometry.
    fn native_swiglu(gate:FloatTensor<Self>,up:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::SwiGluError>;
    /// Actual original selected first-order input VJPs, including intermediate storage rounding.
    fn native_swiglu_backward(gate:FloatTensor<Self>,up:FloatTensor<Self>,gradient:FloatTensor<Self>,selection:NativeSwiGluSelection)
        -> Result<NativeSwiGluBackward<Self>,Self::SwiGluError>;
}
