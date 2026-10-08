//! Frozen original RUDA NF4 expert cubes, with discrete native U32 row assignments.
use crate::{Backend,frozen_nf4::Nf4ProjectionOptions,tensor::{FloatTensor,IntTensor}};
use core::fmt;

/// Actual logical cube and explicitly owned global expert range, without per-expert reblocking.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Nf4GroupedOptions {
    /// Number of actual resident expert matrices.
    pub experts:usize,
    /// First global expert ID represented by this actual local cube.
    pub expert_start:usize,
    /// Per-expert geometry and original execution choices.
    pub projection:Nf4ProjectionOptions,
}
/// Actual source expert payload; frozen FP32 scales/book never become activation storage.
#[derive(Clone,Debug)]
pub struct Nf4ExpertPayload<B:Backend> {
    /// Original high-nibble-first U8 bytes for flattened `[E,N,K]`.
    pub packed:IntTensor<B>,
    /// Original FP32 flat-block scales, including blocks crossing expert boundaries.
    pub scales:FloatTensor<B>,
    /// Original sixteen FP32 codebook values.
    pub codebook:FloatTensor<B>,
    /// Explicit source geometry, ownership and execution choices.
    pub options:Nf4GroupedOptions,
}
/// Native selected expert projection and its real first-order input derivative.
pub trait FrozenNf4GroupedOps:Backend {
    /// Native projection/routing failure or unsupported derivative contract.
    type Nf4GroupedError:fmt::Debug;
    /// Validated original row permutation and actual frozen packed operands.
    type Nf4GroupedState:Clone+Send+fmt::Debug+'static;
    /// Execute one actual assigned expert per incoming row and preserve original row order.
    fn frozen_nf4_grouped_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,payload:Nf4ExpertPayload<Self>)
        -> Result<(FloatTensor<Self>,Self::Nf4GroupedState),Self::Nf4GroupedError>;
    /// Original first-order input VJP, returned in original activation storage.
    fn frozen_nf4_grouped_input_backward(state:Self::Nf4GroupedState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::Nf4GroupedError>;
}
/// Selected original NF4 gate/up/down plus source-native storage-rounded SwiGLU.
pub trait FrozenNf4SwiGluOps:FrozenNf4GroupedOps {
    /// Original native row mapping and optional actual input VJP intermediates.
    type Nf4SwiGluState:Clone+Send+fmt::Debug+'static;
    /// Execute the original selected frozen expert chain. AD always retains the
    /// cache required by actual tracked input, independently of `retain_input`.
    fn frozen_nf4_swiglu_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,gate:Nf4ExpertPayload<Self>,up:Nf4ExpertPayload<Self>,down:Nf4ExpertPayload<Self>,retain_input:bool)
        -> Result<(FloatTensor<Self>,Self::Nf4SwiGluState),Self::Nf4GroupedError>;
    /// Real native first-order input VJP; no base, discrete selection or higher-order gradients.
    fn frozen_nf4_swiglu_input_backward(state:Self::Nf4SwiGluState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::Nf4GroupedError>;
}
