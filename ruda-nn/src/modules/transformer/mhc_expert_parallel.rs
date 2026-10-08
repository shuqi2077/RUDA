use core::fmt;
use ruda_model::{module::Module, tensor::{Tensor, MoeDispatchOps, MoeReceivedOps, VariableTensorCollective, backend::Backend}};
use ruda_autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy};
use crate::expert_parallel::{ExpertParallelMoeLayer, ExpertParallelMoeError, ExpertParallelMoeOutput,
    ExpertParallelGeometry, ExpertParallelReceived, ExpertParallelSwiGluExperts};
use super::{ProjectedFeedForward, TransformerProjectionShape, TransformerProjection, MhcResidualBranchShape, MhcResidualBranch};

/// Original routed expert transport and explicitly present residual-free shared FFN.
/// Communicators belong to the execution call, not the model's Module record or valid() conversion.
#[derive(Module, Debug)]
pub struct ExpertParallelMhcFeedForward<B: Backend, Q: Module<B>, E: Module<B> = ExpertParallelSwiGluExperts<B>> {
    pub routed: ExpertParallelMoeLayer<B, Q, E>,
    pub shared: Option<ProjectedFeedForward<B, Q>>,
}

impl<B: Backend, Q: TransformerProjectionShape<B>, E: ExpertParallelGeometry<B>> ExpertParallelMhcFeedForward<B, Q, E> {
    pub fn from_parts(routed: ExpertParallelMoeLayer<B, Q, E>, shared: Option<ProjectedFeedForward<B, Q>>) -> Self {
        let value = Self { routed, shared };
        value.validate_branch();
        value
    }
}

impl<B: Backend, Q: TransformerProjectionShape<B>, E: ExpertParallelGeometry<B>> MhcResidualBranchShape<B> for ExpertParallelMhcFeedForward<B, Q, E> {
    fn width(&self) -> usize { self.routed.width() }
    fn validate_branch(&self) {
        self.routed.validate();
        if let Some(shared) = &self.shared {
            shared.validate_branch();
            assert_eq!(shared.width(), self.width(), "mHC owned/shared expert branch width differs");
        }
    }
}

#[derive(Debug)]
pub enum ExpertParallelMhcError<C: fmt::Debug, P: fmt::Debug, E: fmt::Debug> {
    Expert(ExpertParallelMoeError<C, P, E>),
    Shared(P),
}
impl<C: fmt::Debug, P: fmt::Debug, E: fmt::Debug> fmt::Display for ExpertParallelMhcError<C, P, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Expert(error) => write!(f, "mHC routed expert: {error}"), Self::Shared(error) => write!(f, "mHC shared expert: {error:?}") }
    }
}
impl<C: fmt::Debug, P: fmt::Debug, E: fmt::Debug> core::error::Error for ExpertParallelMhcError<C, P, E> {}

/// Caller-selected local branch or real rank-owned expert branch in an otherwise unchanged mHC layer.
#[derive(Module, Debug)]
pub enum MixedMhcFeedForward<B: Backend, L: Module<B>, Q: Module<B>, E: Module<B>> {
    Local(L),
    Parallel(ExpertParallelMhcFeedForward<B, Q, E>),
}
impl<B: Backend, L: MhcResidualBranchShape<B>, Q: TransformerProjectionShape<B>, E: ExpertParallelGeometry<B>>
    MhcResidualBranchShape<B> for MixedMhcFeedForward<B, L, Q, E> {
    fn width(&self) -> usize { match self { Self::Local(value) => value.width(), Self::Parallel(value) => value.width() } }
    fn validate_branch(&self) { match self { Self::Local(value) => value.validate_branch(), Self::Parallel(value) => value.validate_branch() } }
}
#[derive(Debug)]
pub enum MixedMhcBranchError<L: fmt::Debug, C: fmt::Debug, P: fmt::Debug, E: fmt::Debug> {
    Local(L),
    Parallel(ExpertParallelMhcError<C, P, E>),
}
impl<L: fmt::Debug, C: fmt::Debug, P: fmt::Debug, E: fmt::Debug> fmt::Display for MixedMhcBranchError<L, C, P, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Local(error) => write!(f, "mHC local branch: {error:?}"), Self::Parallel(error) => write!(f, "{error}") }
    }
}
impl<L: fmt::Debug, C: fmt::Debug, P: fmt::Debug, E: fmt::Debug> core::error::Error for MixedMhcBranchError<L, C, P, E> {}

macro_rules! execute_mhc_experts {
    ($backend:ty, [$($generics:tt)*], $routed:ident, $forward:ident, $detailed:ident) => {
        impl<$($generics)*, Q: TransformerProjection<$backend>, E: ExpertParallelReceived<$backend>>
            ExpertParallelMhcFeedForward<$backend, Q, E> {
            pub fn $forward<C: VariableTensorCollective<B>, const D: usize>(&self, input: Tensor<$backend, D>, communicator: C)
                -> Result<Tensor<$backend, D>, ExpertParallelMhcError<C::Error, Q::Error, E::Error>> {
                self.$detailed(input, communicator).map(|result| result.output)
            }

            /// Original router logits/assignment counts survive without re-running a dropout-bearing router.
            pub fn $detailed<C: VariableTensorCollective<B>, const D: usize>(&self, input: Tensor<$backend, D>, communicator: C)
                -> Result<ExpertParallelMoeOutput<$backend, D>, ExpertParallelMhcError<C::Error, Q::Error, E::Error>> {
                self.validate_branch();
                let mut result = self.routed.$routed(input.clone(), communicator).map_err(ExpertParallelMhcError::Expert)?;
                if let Some(shared) = &self.shared {
                    let output = shared.forward(input).map_err(ExpertParallelMhcError::Shared)?;
                    assert_eq!(output.dims(), result.output.dims(), "mHC owned/shared expert output geometry differs");
                    result.output = result.output + output;
                }
                Ok(result)
            }
        }

        impl<$($generics)*, L: MhcResidualBranch<$backend>, Q: TransformerProjection<$backend>, E: ExpertParallelReceived<$backend>>
            MixedMhcFeedForward<$backend, L, Q, E> {
            /// Lazily obtain the actual communicator only for parallel layers; no dummy local transport is needed.
            pub fn $forward<C: VariableTensorCollective<B>, G: FnOnce() -> C>(&self, input: Tensor<$backend, 3>, communicator: G)
                -> Result<Tensor<$backend, 3>, MixedMhcBranchError<L::Error, C::Error, Q::Error, E::Error>> {
                match self {
                    Self::Local(value) => value.forward_branch(input).map_err(MixedMhcBranchError::Local),
                    Self::Parallel(value) => value.$forward(input, communicator()).map_err(MixedMhcBranchError::Parallel),
                }
            }
        }
    };
}
execute_mhc_experts!(B, [B: MoeDispatchOps + MoeReceivedOps], forward_detailed_inference, forward_inference, forward_detailed_inference);
execute_mhc_experts!(Autodiff<B, S>, [B: MoeDispatchOps + MoeReceivedOps, S: CheckpointStrategy], forward_detailed, forward, forward_detailed);
