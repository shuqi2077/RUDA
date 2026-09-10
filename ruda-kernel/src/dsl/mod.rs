pub use ruda_core::tensor as zspace;
use ruda_core::tensor::Shape;
use ruda_core::tensor::Strides;

/// Ruda Frontend Types.
pub mod frontend;
/// Input Output utilities.
pub mod io;

pub mod post_processing;

/// Some future utilities that work across environments.
pub use ruda_core::future;

use ruda_core::ir::VectorSize;
use ruda::runtime::client::ComputeClient;
pub use ruda::runtime::memory_management::MemoryConfiguration;
use ruda::runtime::server::RudaCountSelection;
pub use frontend::cmma;

/// Ruda Language Internal Representation.
pub use ruda_core::ir as ir;

pub mod codegen;
pub mod compute;
pub mod prelude;

mod pod;

pub use codegen::*;
pub use ruda::runtime::backend::*;
pub use pod::*;

pub use ruda_kernel_macros::{
    AutotuneKey, RudaLaunch, RudaType, RudaTypeMut, comment, comptime,
    comptime_type, ruda, derive_ruda_comptime, derive_expand, intrinsic, terminate,
};

pub use ruda::runtime::benchmark;
pub use ruda::runtime::client;
pub use ruda::runtime::compiler::{CompilationError, Compiler, RudaTask};
pub use ruda::runtime::memory_management::MemoryUsage;
pub use ruda::runtime::server;
pub use ruda::runtime::tune;

use frontend::LaunchArg;

pub use ruda_core::*;

pub use prelude::RudaCount;
pub use prelude::{RudaDim, ExecutionMode};

pub use num_traits;

mod id;
pub use id::*;

// Private utils for macros
#[doc(hidden)]
pub mod __private {
    pub use alloc::{format, vec};
    pub use paste::paste;
}

pub use prelude::{Assign, IntoRuntime};

/// Calculate the number of rudas required to execute an operation where one ruda unit is
/// assigned to one element.
pub fn calculate_ruda_count_elemwise<R: Runtime>(
    client: &ComputeClient<R>,
    num_elems: usize,
    ruda_dim: RudaDim,
) -> RudaCount {
    if num_elems == 0 {
        return RudaCount::Static(0, 0, 0);
    }
    let num_rudas = num_elems.div_ceil(ruda_dim.num_elems() as usize);
    RudaCountSelection::new(client, num_rudas as u32).ruda_count()
}

pub fn tensor_vectorization_factor(
    factors: &[VectorSize],
    shape: &Shape,
    strides: &Strides,
    dim: usize,
) -> VectorSize {
    tensor_vector_size_parallel(factors.iter().cloned(), shape, strides, dim)
}
pub fn tensor_vectorization(
    factors: &[VectorSize],
    shape: &Shape,
    strides: &Strides,
    dim: usize,
) -> VectorSize {
    tensor_vector_size_parallel(factors.iter().cloned(), shape, strides, dim)
}

#[derive(Debug, Clone)]
pub enum VectorizationError {
    AxisOutOfBounds,
    StrideMismatch,
    NoValidVectorization,
}

/// Find the maximum vector size usable for parallel vectorization along the given axis
/// from the supported vector sizes or return 1 if vectorization is impossible.
///
/// This function is designed to never return a vector size above 1 by error,
/// but doesn't guarantee to always return the actual maximum possible vector size.
/// That is, it may be overly strict.
///
/// Currently, this checks that the stride of the axis is 1, that its shape is
/// divisible by a candidate vector size and that every non-broadcast stride outside
/// the axis is divisible by the vector size.
/// The last condition ensures a vectorized read on `axis` stays contiguous in the
/// source buffer as coordinates in other dimensions change.
pub fn tensor_vector_size_parallel(
    optimized_vector_sizes: impl Iterator<Item = VectorSize>,
    shape: &Shape,
    strides: &Strides,
    axis: usize,
) -> VectorSize {
    try_tensor_vector_size_parallel(optimized_vector_sizes, shape, strides, axis).unwrap_or(1)
}

/// Like `try_tensor_vector_size_parallel` but does not assume 1 is supported
pub fn try_tensor_vector_size_parallel(
    supported_vector_sizes: impl Iterator<Item = VectorSize>,
    shape: &Shape,
    strides: &Strides,
    axis: usize,
) -> Result<VectorSize, VectorizationError> {
    let stride = strides
        .get(axis)
        .ok_or(VectorizationError::AxisOutOfBounds)?;
    if *stride != 1 {
        return Err(VectorizationError::StrideMismatch);
    }

    let axis_shape = shape.get(axis).ok_or(VectorizationError::AxisOutOfBounds)?;

    // Check all non-axis strides. Stride 0 is a broadcast and
    // never contributes to the source offset, so it can be ignored. Every other
    // dim can shift the source offset when its coord changes, so its stride must
    // be a multiple of the vector size for vectorized reads to stay aligned.
    // Unit-size dims are included for simplicity; they only cause false negatives
    // (vectorization disabled) rather than incorrect output.
    supported_vector_sizes
        .filter(|&vector_size| {
            vector_size != 0
                && axis_shape % vector_size == 0
                && strides
                    .iter()
                    .enumerate()
                    .all(|(i, &stride)| i == axis || stride % vector_size == 0)
        })
        .max()
        .ok_or(VectorizationError::NoValidVectorization)
}

/// Find the maximum vector size usable for perpendicular vectorization along the given axis
/// from the supported vector sizes or return 1 if vectorization is impossible.
///
/// This function is designed to never return a vector size above 1 by error,
/// but doesn't guarantee to always return the actual maximum possible vector size.
/// That is, it may be overly strict.
///
/// Checks that smaller-stride axes form a contiguous span ending at the axis stride,
/// and that the axis stride and every larger stride are divisible by the candidate vector size.
pub fn tensor_vector_size_perpendicular(
    supported_vector_sizes: impl Iterator<Item = VectorSize>,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
) -> VectorSize {
    try_tensor_vector_sizes_perpendicular(supported_vector_sizes, shape, strides, axis).unwrap_or(1)
}

/// Like `tensor_vector_sizes_perpendicular` but does not assume 1 is supported
pub fn try_tensor_vector_sizes_perpendicular(
    supported_vector_sizes: impl Iterator<Item = VectorSize>,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
) -> Result<VectorSize, VectorizationError> {
    let axis_stride = *strides
        .get(axis)
        .ok_or(VectorizationError::AxisOutOfBounds)?;
    shape.get(axis).ok_or(VectorizationError::AxisOutOfBounds)?;
    if shape.len() != strides.len() || axis_stride == 0 {
        return Err(VectorizationError::StrideMismatch);
    }

    let mut inner_axes = strides
        .iter()
        .zip(shape.iter())
        .filter_map(|(&stride, &size)| {
            (stride < axis_stride && size != 1).then_some((stride, size))
        })
        .collect::<alloc::vec::Vec<_>>();
    inner_axes.sort_unstable_by_key(|&(stride, _)| stride);

    let mut span = 1usize;
    for (stride, size) in inner_axes {
        if stride != span {
            return Err(VectorizationError::StrideMismatch);
        }
        span = span.checked_mul(size).ok_or(VectorizationError::StrideMismatch)?;
    }
    if axis_stride != span {
        return Err(VectorizationError::StrideMismatch);
    }

    supported_vector_sizes
        .filter(|&vector_size| {
            vector_size != 0
                && axis_stride % vector_size == 0
                && strides.iter().all(|&stride| stride < axis_stride || stride % vector_size == 0)
        })
        .max()
        .ok_or(VectorizationError::NoValidVectorization)
}

/// Runtime arguments to launch a kernel.
pub type RuntimeArg<T, R> = <T as LaunchArg>::RuntimeArg<R>;
pub type ExpandType<T> = <T as crate::dsl::prelude::RudaType>::ExpandType;

#[cfg(feature = "frontend-tests")]
/// Tests only useful for runtimes.
pub mod runtime_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn try_parallel(
        sizes: &[VectorSize],
        shape: &[usize],
        strides: &[usize],
        axis: usize,
    ) -> Result<VectorSize, VectorizationError> {
        try_tensor_vector_size_parallel(
            sizes.iter().copied(),
            &Shape::from(shape.iter().copied()),
            &Strides::new(strides),
            axis,
        )
    }

    #[test]
    fn parallel_contiguous_picks_max_vector_size() {
        // Contiguous [1, 9, 4], vectorize along last dim (stride 1).
        // Outer stride 4 is a multiple of 4, so vec_size = 4 is safe.
        let v = try_parallel(&[1, 2, 4], &[1, 9, 4], &[36, 4, 1], 2).unwrap();
        assert_eq!(v, 4);
    }

    #[test]
    fn parallel_unfold_step_one_rejects_vectorization() {
        // Unfold view produced by `unfold(1, 4, 1)` on a [1, 12] contiguous tensor:
        // shape [1, 9, 4], strides [12, 1, 1]. The frame dim has stride 1, so each
        // step in the frame coord shifts the source offset by 1 - not a multiple
        // of any vec_size > 1, so vectorized reads would be unaligned and return
        // the wrong data. Must fall back to vec_size = 1.
        let v = try_parallel(&[1, 2, 4], &[1, 9, 4], &[12, 1, 1], 2).unwrap();
        assert_eq!(v, 1);
    }

    #[test]
    fn parallel_unfold_step_two_allows_vectorization() {
        // Same unfold pattern but with step=2: strides [12, 2, 1]. Frame coord
        // shifts source by 2 (still not a multiple of 4), so vec_size = 4 must
        // be rejected - but vec_size = 2 is fine.
        let v = try_parallel(&[1, 2, 4], &[1, 9, 4], &[12, 2, 1], 2).unwrap();
        assert_eq!(v, 2);
    }

    #[test]
    fn parallel_broadcast_dim_ignored() {
        // Broadcast dim has stride 0; it never shifts the source offset, so
        // it should not disqualify vectorization.
        let v = try_parallel(&[1, 2, 4], &[1, 9, 4], &[0, 4, 1], 2).unwrap();
        assert_eq!(v, 4);
    }

    #[test]
    fn parallel_axis_stride_not_one_is_error() {
        let err = try_parallel(&[1, 2, 4], &[1, 9, 4], &[36, 1, 4], 2).unwrap_err();
        assert!(matches!(err, VectorizationError::StrideMismatch));
    }
}

pub use crate::{unexpanded, expand_error, expand_assert, size, define, debug_print, debug_print_expand, define_scalar, define_size};

#[cfg(feature = "frontend-tests")]
pub use crate::{testgen_all_reduce, testgen_assign, testgen_atomic_untyped, testgen_atomic_int, testgen_atomic_float, testgen_barrier, testgen_binary, testgen_binary_untyped, testgen_branch, testgen_cluster, testgen_cmma, testgen_comparison, testgen_const_match, testgen_constants, testgen_debug, testgen_different_rank, testgen_enums, testgen_file, testgen_index, testgen_launch, testgen_launch_untyped, testgen_metadata, testgen_minifloat, testgen_all, testgen_float, testgen_int, testgen_uint, testgen_untyped, as_bytes, as_type, testgen_numeric, testgen_plane, testgen_properties, testgen_saturating_uint, testgen_saturating_int, testgen_sequence, testgen_slice, testgen_stream, testgen_sync_plane, testgen_tensor_indexing, testgen_tensormap, testgen_to_client, testgen_topology, testgen_unary, testgen_unary_int, testgen_unroll, testgen_vector};

pub mod lowering;
