use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

/// Operation family trait for cumulative operations
pub trait CumulativeOpFamily: Send + Sync + 'static {
    type CumulativeOp<C: Numeric>: CumulativeOp<C>;
}

/// Trait for cumulative operations
#[ruda]
pub trait CumulativeOp<C: Numeric>: 'static + Send + Sync {
    /// Execute a cumulative operation
    fn execute(lhs: C, rhs: C) -> C;

    /// Get the initial value for the accumulator
    fn init_value(first_element: C) -> C;
}

// Operation types
pub(super) struct SumOp;
pub(super) struct ProdOp;
pub(super) struct MaxOp;
pub(super) struct MinOp;

// Implement CumulativeOpFamily for each operation
impl CumulativeOpFamily for SumOp {
    type CumulativeOp<C: Numeric> = Self;
}

impl CumulativeOpFamily for ProdOp {
    type CumulativeOp<C: Numeric> = Self;
}

impl CumulativeOpFamily for MaxOp {
    type CumulativeOp<C: Numeric> = Self;
}

impl CumulativeOpFamily for MinOp {
    type CumulativeOp<C: Numeric> = Self;
}

// Implement CumulativeOp for each operation type
#[ruda]
impl<N: Numeric> CumulativeOp<N> for SumOp {
    fn execute(lhs: N, rhs: N) -> N {
        lhs + rhs
    }

    fn init_value(_first_element: N) -> N {
        N::zero()
    }
}

#[ruda]
impl<N: Numeric> CumulativeOp<N> for ProdOp {
    fn execute(lhs: N, rhs: N) -> N {
        lhs * rhs
    }

    fn init_value(_first_element: N) -> N {
        N::from_int(1)
    }
}

#[ruda]
impl<N: Numeric> CumulativeOp<N> for MaxOp {
    fn execute(lhs: N, rhs: N) -> N {
        max(lhs, rhs)
    }

    fn init_value(first_element: N) -> N {
        first_element
    }
}

#[ruda]
impl<N: Numeric> CumulativeOp<N> for MinOp {
    fn execute(lhs: N, rhs: N) -> N {
        min(lhs, rhs)
    }

    fn init_value(first_element: N) -> N {
        first_element
    }
}

