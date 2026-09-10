//! Operators shared by device, block and logical-warp collectives.

use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

pub(crate) mod merge;
pub mod radix;
pub mod record;
pub mod decompose;
pub mod tile;

#[ruda]
pub trait RudaUnaryOp<T: RudaType, U: RudaType>: RudaType {
    fn apply(&self, value: T) -> U;
}

/// A binary operator. Scans and reductions require associativity and preserve
/// operand order; adjacent-difference operations do not require associativity.
#[ruda]
pub trait RudaBinaryOp<T: RudaType>: RudaType {
    fn combine(&self, left: T, right: T) -> T;
}

/// A strict weak ordering for comparison-based algorithms.
#[ruda]
pub trait RudaCompare<T: RudaType>: RudaType {
    fn before(&self, left: T, right: T) -> bool;
}

/// An equivalence relation for adjacent-key operations.
#[ruda]
pub trait RudaKeyEqual<T: RudaType>: RudaType {
    fn equal(&self, left: T, right: T) -> bool;
}

#[derive(Clone, Copy, Debug, RudaType, RudaLaunch)]
pub struct RudaSum;

#[ruda]
impl<T: Numeric> RudaBinaryOp<T> for RudaSum {
    fn combine(&self, left: T, right: T) -> T {
        left + right
    }
}

#[derive(Clone, Copy, Debug, RudaType, RudaLaunch)]
pub struct RudaProduct;

#[ruda]
impl<T: Numeric> RudaBinaryOp<T> for RudaProduct {
    fn combine(&self, left: T, right: T) -> T {
        left * right
    }
}

#[derive(Clone, Copy, Debug, RudaType, RudaLaunch)]
pub struct RudaSubtract;

#[ruda]
impl<T: Numeric> RudaBinaryOp<T> for RudaSubtract {
    fn combine(&self, left: T, right: T) -> T { left - right }
}

#[derive(Clone, Copy, Debug, RudaType, RudaLaunch)]
pub struct RudaMinimum;

#[ruda]
impl<T: Numeric> RudaBinaryOp<T> for RudaMinimum {
    fn combine(&self, left: T, right: T) -> T {
        if right < left { right } else { left }
    }
}

#[derive(Clone, Copy, Debug, RudaType, RudaLaunch)]
pub struct RudaMaximum;

#[ruda]
impl<T: Numeric> RudaBinaryOp<T> for RudaMaximum {
    fn combine(&self, left: T, right: T) -> T {
        if left < right { right } else { left }
    }
}

#[derive(Clone, Copy, Debug, RudaType, RudaLaunch)]
pub struct RudaAscending;

#[ruda]
impl<T: Numeric> RudaCompare<T> for RudaAscending {
    fn before(&self, left: T, right: T) -> bool {
        left < right
    }
}

#[derive(Clone, Copy, Debug, RudaType, RudaLaunch)]
pub struct RudaDescending;

#[ruda]
impl<T: Numeric> RudaCompare<T> for RudaDescending {
    fn before(&self, left: T, right: T) -> bool {
        left > right
    }
}

#[derive(Clone, Copy, Debug, RudaType, RudaLaunch)]
pub struct RudaEqual;

macro_rules! clone_operator_launch {
    ($($name:ident),* $(,)?) => {$(
        impl<R: Runtime> Clone for $name<R> {
            fn clone(&self) -> Self { Self::new() }
        }
    )*};
}

clone_operator_launch!(RudaSumLaunch, RudaProductLaunch, RudaSubtractLaunch,
    RudaMinimumLaunch, RudaMaximumLaunch, RudaAscendingLaunch, RudaDescendingLaunch,
    RudaEqualLaunch);

#[ruda]
impl<T: Scalar> RudaKeyEqual<T> for RudaEqual {
    fn equal(&self, left: T, right: T) -> bool {
        left == right
    }
}
