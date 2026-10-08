//! Native local MoE training contracts over existing device routing and expert kernels.
use crate::{Backend,tensor::{FloatTensor,IntTensor}};
use core::fmt;

/// Original continuous weight scoring, independent of discrete expert selection.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum MoeRouterScoring {
    /// FP32 softmax over every expert before selecting continuous weights.
    Softmax,
    /// Original pointwise stable sigmoid scores.
    Sigmoid,
}
/// Explicit original continuous routing-weight policy; no model family or default is inferred.
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct MoeRouterWeightOptions {
    /// Original full-softmax or pointwise-sigmoid scoring.
    pub scoring:MoeRouterScoring,
    /// Normalize the actual selected slots, retaining duplicate-slot gather semantics.
    pub renormalize:bool,
    /// Finite positive FP32 multiplier, applied after selected normalization.
    pub scale:f32,
}
/// Original discrete expert selection. Correction bias affects selection only.
#[derive(Clone,Copy,Debug,PartialEq)]
pub enum MoeSelectionOptions {
    /// Original full softmax/top-k, with lower expert IDs resolving exact ties.
    Softmax {
        /// Actual selected experts per token.
        top_k:usize,
        /// Original inference selection-weight normalization; training weights are explicit separately.
        renormalize:bool,
    },
    /// Original group-limited sigmoid selection with optional FP32 correction bias.
    SigmoidGrouped {
        /// Actual selected experts per token.
        top_k:usize,
        /// Actual equal expert-group count.
        groups:usize,
        /// Actual selected group count.
        selected_groups:usize,
        /// Use the sum of the two highest corrected scores rather than the group maximum.
        group_top_two:bool,
        /// Original inference selected-weight normalization.
        renormalize:bool,
        /// Original finite positive inference weight multiplier.
        scale:f32,
    },
}
/// Original segmented expert GEMM strategy, selected independently for forward and backward.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum MoeExpertStrategy {
    /// Original FP32-accumulating device scalar kernel, not host computation.
    Scalar,
    /// Original capability-only choice; failed compilation/launch never selects recovery fallback.
    Auto,
    /// Require original supported half/BF16 cooperative matrix kernels.
    TensorCore,
}
/// Original routing-weight VJP reduction order.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum MoeCombineGradientStrategy {
    /// Original serial per-slot reduction order.
    Serial,
    /// Explicit original full-plane reduction, requiring supported hardware/grid.
    Plane,
}
/// Actual complete local expert execution choices, with no implicit tuning or retry policy.
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct MoeOptions {
    /// Original actual discrete selection policy.
    pub selection:MoeSelectionOptions,
    /// Original actual continuous selected-weight policy, independent of correction bias.
    pub weights:MoeRouterWeightOptions,
    /// Original actual expert forward strategy.
    pub forward:MoeExpertStrategy,
    /// Original actual expert backward strategy.
    pub backward:MoeExpertStrategy,
    /// Original actual combine weight-gradient reduction strategy.
    pub combine_backward:MoeCombineGradientStrategy,
}
/// Actual original first-order derivatives of the native routed expert branch.
#[derive(Debug)]
pub struct MoeBackward<B:Backend> {
    /// Input gradient after dispatch-copy backward, without a second routing-weight multiplication.
    pub input:FloatTensor<B>,
    /// Original source-logits storage VJP, including nonselected softmax experts.
    pub logits:FloatTensor<B>,
    /// Original FP32 gate-weight gradients.
    pub gate:FloatTensor<B>,
    /// Original FP32 up-weight gradients.
    pub up:FloatTensor<B>,
    /// Original FP32 down-weight gradients.
    pub down:FloatTensor<B>,
}
/// Actual first-order derivatives requested by the caller or tracked AD parents.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct MoeGradientSelection {
    /// Preserve upstream input derivatives independently of expert-weight trainability.
    pub input:bool,
    /// Original fixed-selection logits derivative.
    pub logits:bool,
    /// Original FP32 gate cube derivative.
    pub gate:bool,
    /// Original FP32 up cube derivative.
    pub up:bool,
    /// Original FP32 down cube derivative.
    pub down:bool,
}
/// Only requested original native derivatives; None is absence, never a synthetic zero tensor.
#[derive(Debug)]
pub struct MoeBackwardSelected<B:Backend> {
    /// Requested source-token input derivative.
    pub input:Option<FloatTensor<B>>,
    /// Requested source-logit derivative.
    pub logits:Option<FloatTensor<B>>,
    /// Requested original FP32 gate cube derivative.
    pub gate:Option<FloatTensor<B>>,
    /// Requested original FP32 up cube derivative.
    pub up:Option<FloatTensor<B>>,
    /// Requested original FP32 down cube derivative.
    pub down:Option<FloatTensor<B>>,
}
/// Native execution failure or an unsupported derivative of the first-order native kernels.
#[derive(Debug)]
pub enum MoeAutodiffError<E:fmt::Debug> {
    /// Original native validation/launch error.
    Native(E),
    /// Native router/expert VJPs do not provide higher derivatives.
    HigherDerivativeUnsupported,
}
impl<E:fmt::Debug> fmt::Display for MoeAutodiffError<E> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Native(error)=>write!(f,"native MoE: {error:?}"),Self::HigherDerivativeUnsupported=>f.write_str("native MoE provides first-order derivatives only")}
    }
}
impl<E:fmt::Debug> core::error::Error for MoeAutodiffError<E> {}

/// Optional actual device MoE extension. No dense host expert evaluation is provided.
pub trait MoeOps:Backend {
    /// Original native or first-order contract error.
    type MoeError:fmt::Debug;
    /// Actual opaque forward/dispatch state; contains original native handles, not host model copies.
    type MoeState:Clone+Send+fmt::Debug+'static;
    /// FP32 continuous weights for explicitly supplied integer selections. Invalid
    /// indices retain the original bounds-safe whole-row NaN semantics.
    fn moe_selected_weights(logits:FloatTensor<Self>,indices:IntTensor<Self>,options:MoeRouterWeightOptions) -> Result<FloatTensor<Self>,Self::MoeError>;
    /// Original source-logits storage VJP for fixed selections; gradient weights are FP32.
    fn moe_selected_weights_backward(logits:FloatTensor<Self>,indices:IntTensor<Self>,gradient:FloatTensor<Self>,options:MoeRouterWeightOptions)
        -> Result<FloatTensor<Self>,Self::MoeError>;
    /// Original top-k/group selection -> FP32 continuous weights -> native dispatch
    /// -> original SwiGLU experts -> ordered combine. No tokens are capacity-dropped.
    fn moe_forward(input:FloatTensor<Self>,logits:FloatTensor<Self>,correction_bias:Option<FloatTensor<Self>>,
        gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,options:MoeOptions) -> Result<(FloatTensor<Self>,Self::MoeState),Self::MoeError>;
    /// Same original forward with an explicit backward-output requirement. A native
    /// implementation may omit expert activation caches when no expert/input VJP is
    /// required. Requesting an unavailable expert VJP from that state must error.
    /// The compatibility default retains the original complete forward state.
    fn moe_forward_selected(input:FloatTensor<Self>,logits:FloatTensor<Self>,correction_bias:Option<FloatTensor<Self>>,
        gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,options:MoeOptions,selection:MoeGradientSelection)
        -> Result<(FloatTensor<Self>,Self::MoeState),Self::MoeError> {
        let _=selection;Self::moe_forward(input,logits,correction_bias,gate,up,down,options)
    }
    /// Same original continuous weight policy and native expert output, without retained backward caches.
    fn moe_inference(input:FloatTensor<Self>,logits:FloatTensor<Self>,correction_bias:Option<FloatTensor<Self>>,
        gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,options:MoeOptions) -> Result<FloatTensor<Self>,Self::MoeError>;
    /// Actual original discrete U32 expert IDs, with no host readback or differentiable selection claim.
    fn moe_route_indices(state:&Self::MoeState) -> IntTensor<Self>;
    /// Original first-order input/logits/expert derivatives. Expert gradients retain
    /// native FP32 accumulation/output; ordinary AD casts them at the parent-storage boundary.
    fn moe_backward(state:Self::MoeState,gradient:FloatTensor<Self>) -> Result<MoeBackward<Self>,Self::MoeError>;
    /// Explicit requested VJP outputs. Device implementations can omit unneeded native launches
    /// and allocations; the compatibility default retains exact full-backward behavior.
    fn moe_backward_selected(state:Self::MoeState,gradient:FloatTensor<Self>,selection:MoeGradientSelection)
        -> Result<MoeBackwardSelected<Self>,Self::MoeError> {
        let result=Self::moe_backward(state,gradient)?;
        Ok(MoeBackwardSelected {input:selection.input.then_some(result.input),logits:selection.logits.then_some(result.logits),
            gate:selection.gate.then_some(result.gate),up:selection.up.then_some(result.up),down:selection.down.then_some(result.down)})
    }
}
