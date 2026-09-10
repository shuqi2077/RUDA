use crate::device::optim::{
    elemwise::{ElemwiseOptimization, ElemwiseOptimizationState},
    matmul::{MatmulOptimization, MatmulOptimizationState},
    reduce::{ReduceOptimization, ReduceOptimizationState},
    reduce_broadcasted::{ReduceBroadcastedOptimization, ReduceBroadcastedOptimizationState},
};
use ruda_kernel::dsl::Runtime;
use serde::{Deserialize, Serialize};

/// Fusion optimization type for ruda.
///
/// More optimization variants should be added here.
#[allow(clippy::large_enum_variant)]
pub enum RudaOptimization<R: Runtime> {
    ElementWise(ElemwiseOptimization<R>),
    Matmul(MatmulOptimization<R>),
    Reduce(ReduceOptimization<R>),
    ReduceBroadcasted(ReduceBroadcastedOptimization<R>),
}

impl<R: Runtime> core::fmt::Debug for RudaOptimization<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.to_opt_state();
        f.write_fmt(format_args!("{value:?}"))
    }
}

impl<R: Runtime> RudaOptimization<R> {
    /// Serializes the current optimization to its state.
    pub fn to_opt_state(&self) -> RudaOptimizationState {
        match self {
            Self::ElementWise(value) => RudaOptimizationState::ElementWise(value.to_state()),
            Self::Matmul(value) => RudaOptimizationState::Matmul(value.to_state()),
            Self::Reduce(value) => RudaOptimizationState::Reduce(value.to_state()),
            Self::ReduceBroadcasted(value) => {
                RudaOptimizationState::ReduceBroadcasted(value.to_state())
            }
        }
    }
}

impl<R: Runtime> crate::NumOperations for RudaOptimization<R> {
    fn len(&self) -> usize {
        match self {
            Self::ElementWise(op) => op.num_ops_fused(),
            Self::Matmul(op) => op.num_ops_fused(),
            Self::Reduce(op) => op.num_ops_fused(),
            Self::ReduceBroadcasted(op) => op.num_ops_fused(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            RudaOptimization::ElementWise(..) => "ElementWise",
            RudaOptimization::Matmul(..) => "Matmul",
            RudaOptimization::Reduce(..) => "Reduce",
            RudaOptimization::ReduceBroadcasted(..) => "ReduceBroadcasted",
        }
    }
}

/// Fusion optimization state type for ruda.
///
/// More optimization variants should be added here.
#[allow(clippy::large_enum_variant)]
#[derive(Serialize, Deserialize, Debug)]
pub enum RudaOptimizationState {
    ElementWise(ElemwiseOptimizationState),
    Matmul(MatmulOptimizationState),
    Reduce(ReduceOptimizationState),
    ReduceBroadcasted(ReduceBroadcastedOptimizationState),
}
