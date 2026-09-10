//! Reduction operations for FlexTensor.
//!
//! Optimized with:
//! - Strided iteration (no copy for non-contiguous tensors)
//! - Portable SIMD via macerator (NEON, AVX2, SIMD128, scalar fallback)
//! - Rayon parallelism for large tensors

use alloc::vec;
use alloc::vec::Vec;
use ruda_core::tensor::{DType, element::Element};
use ruda_core::{bytes::Bytes, tensor::Shape};
use half::{bf16, f16};

use ruda_core::tensor::host::strided_index::StridedIter;
use ruda_core::tensor::host::{HostTensor, Layout};

use ruda_core::tensor::host::dtype::{INDEX_DTYPE, float_storage_as_f32};

/// Assert that a dimension size fits in `isize`, which is required for index-producing
/// operations (argmax, argmin, *_with_indices) that store dimension indices as `isize`.
#[inline(always)]
fn assert_dim_fits_isize(dim_size: usize, dim: usize) {
    assert!(
        dim_size <= isize::MAX as usize,
        "dimension {dim} has size {dim_size} which exceeds isize::MAX"
    );
}

#[cfg(feature = "simd")]
use crate::simd::kernels;

#[cfg(feature = "simd")]
use crate::simd::aligned;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Truncate an i64 to a smaller Pod type, keeping the low-order bytes.
/// Endian-safe: works correctly on both little-endian and big-endian targets.
fn truncate_i64_to_pod<E: bytemuck::Pod>(value: i64) -> E {
    let bytes = value.to_ne_bytes();
    let size = core::mem::size_of::<E>();
    debug_assert!(size <= core::mem::size_of::<i64>());
    let offset = if cfg!(target_endian = "big") {
        core::mem::size_of::<i64>() - size
    } else {
        0
    };
    bytemuck::pod_read_unaligned(&bytes[offset..offset + size])
}

// ============================================================================
// Sum (all elements)
// ============================================================================

/// Sum all elements in a tensor, returning a scalar tensor.
pub fn sum(tensor: HostTensor) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => sum_f32(&tensor),
        DType::F64 => sum_impl::<f64>(&tensor),
        DType::F16 => reduce_scalar_half(&tensor, |a, b| a + b, 0.0, f16::to_f32, f16::from_f32),
        DType::BF16 => reduce_scalar_half(&tensor, |a, b| a + b, 0.0, bf16::to_f32, bf16::from_f32),
        DType::I8 => sum_impl_widening::<i8>(&tensor),
        DType::I16 => sum_impl_widening::<i16>(&tensor),
        DType::I32 => sum_impl_widening::<i32>(&tensor),
        DType::I64 => sum_impl::<i64>(&tensor),
        DType::U8 => sum_impl_widening::<u8>(&tensor),
        DType::U16 => sum_impl_widening::<u16>(&tensor),
        DType::U32 => sum_impl_widening::<u32>(&tensor),
        DType::U64 => sum_impl::<u64>(&tensor),
        _ => panic!("sum: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Optimized f32 sum with SIMD and parallelism.
fn sum_f32(tensor: &HostTensor) -> HostTensor {
    let result = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[f32] = tensor.storage();
            let slice = &data[start..end];
            sum_f32_contiguous(slice)
        }
        None => {
            // Non-contiguous: check if we can sum the buffer directly.
            // For transposed tensors that use all elements (no slicing),
            // the sum is the same regardless of element order.
            let data: &[f32] = tensor.storage();
            let elem_count = tensor.layout().num_elements();

            if data.len() == elem_count {
                // Tensor uses entire buffer - sum directly (order doesn't matter for sum)
                sum_f32_contiguous(data)
            } else {
                // Sliced or partial view - must use strided iteration
                StridedIter::new(tensor.layout()).map(|idx| data[idx]).sum()
            }
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    HostTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::F32)
}

/// SIMD + parallel sum for contiguous f32 slice.
///
/// The parallel threshold is higher than the general `PARALLEL_THRESHOLD` because
/// sum is memory-bound and L2-resident data (< ~16 MiB / 4M f32 elements on
/// Apple M-series) doesn't benefit from rayon's task dispatch overhead.
#[inline]
fn sum_f32_contiguous(data: &[f32]) -> f32 {
    #[cfg(feature = "rayon")]
    if data.len() >= 4 * 1024 * 1024 {
        return sum_f32_parallel(data);
    }

    #[cfg(feature = "simd")]
    {
        kernels::sum_f32(data)
    }

    #[cfg(not(feature = "simd"))]
    {
        data.iter().copied().sum()
    }
}

/// Parallel sum using rayon with SIMD per chunk.
#[cfg(feature = "rayon")]
#[inline]
fn sum_f32_parallel(data: &[f32]) -> f32 {
    const CHUNK_SIZE: usize = 64 * 1024; // 64K elements per chunk

    data.par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            #[cfg(feature = "simd")]
            {
                kernels::sum_f32(chunk)
            }
            #[cfg(not(feature = "simd"))]
            {
                chunk.iter().copied().sum::<f32>()
            }
        })
        .sum()
}

fn sum_impl<E: Element + bytemuck::Pod + Default + core::iter::Sum>(
    tensor: &HostTensor,
) -> HostTensor {
    let result: E = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[E] = tensor.storage();
            data[start..end].iter().copied().sum()
        }
        None => {
            let data: &[E] = tensor.storage();
            StridedIter::new(tensor.layout()).map(|idx| data[idx]).sum()
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    HostTensor::new(
        bytes,
        Layout::contiguous(Shape::from(vec![1])),
        tensor.dtype(),
    )
}

/// Widening scalar reduction for small integer types: accumulate in i64 to avoid overflow.
macro_rules! widening_scalar_reduce {
    ($name:ident, $fold:expr, $init:expr) => {
        fn $name<E>(tensor: &HostTensor) -> HostTensor
        where
            E: Element + bytemuck::Pod + Default,
            i64: From<E>,
        {
            let total: i64 = match tensor.layout().contiguous_offsets() {
                Some((start, end)) => {
                    let data: &[E] = tensor.storage();
                    data[start..end]
                        .iter()
                        .fold($init, |acc, x| ($fold)(acc, i64::from(*x)))
                }
                None => {
                    let data: &[E] = tensor.storage();
                    StridedIter::new(tensor.layout())
                        .fold($init, |acc, idx| ($fold)(acc, i64::from(data[idx])))
                }
            };
            // Truncate back to target type (wrapping, matches PyTorch)
            let result: E = truncate_i64_to_pod(total);
            let bytes = Bytes::from_elems(vec![result]);
            HostTensor::new(
                bytes,
                Layout::contiguous(Shape::from(vec![1])),
                tensor.dtype(),
            )
        }
    };
}

widening_scalar_reduce!(
    sum_impl_widening,
    |acc: i64, x: i64| acc.wrapping_add(x),
    0i64
);
widening_scalar_reduce!(
    prod_impl_widening,
    |acc: i64, x: i64| acc.wrapping_mul(x),
    1i64
);

/// Scalar reduction for half-precision types, accumulating in f32.
fn reduce_scalar_half<E>(
    tensor: &HostTensor,
    fold: fn(f32, f32) -> f32,
    init: f32,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> HostTensor
where
    E: Element + bytemuck::Pod,
{
    let result: f32 = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[E] = tensor.storage();
            data[start..end]
                .iter()
                .fold(init, |acc, x| fold(acc, to_f32(*x)))
        }
        None => {
            let data: &[E] = tensor.storage();
            StridedIter::new(tensor.layout()).fold(init, |acc, idx| fold(acc, to_f32(data[idx])))
        }
    };

    let bytes = Bytes::from_elems(vec![from_f32(result)]);
    HostTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), E::dtype())
}

// ============================================================================
// Sum along dimension
// ============================================================================

/// Sum along a dimension, keeping the dimension with size 1.
pub fn sum_dim(tensor: HostTensor, dim: usize) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => reduce_dim_f32(&tensor, dim, ReduceOp::Sum),
        DType::F64 => reduce_dim_impl::<f64, _>(&tensor, dim, 0.0, |acc, x| acc + x),
        DType::F16 => reduce_dim_half(
            &tensor,
            dim,
            0.0,
            |acc, x| acc + x,
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => reduce_dim_half(
            &tensor,
            dim,
            0.0,
            |acc, x| acc + x,
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I8 => reduce_dim_widening::<i8, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::I16 => reduce_dim_widening::<i16, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::I32 => reduce_dim_widening::<i32, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::I64 => reduce_dim_impl::<i64, _>(&tensor, dim, 0, |acc, x| acc + x),
        DType::U8 => reduce_dim_widening::<u8, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::U16 => reduce_dim_widening::<u16, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::U32 => reduce_dim_widening::<u32, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::U64 => reduce_dim_impl::<u64, _>(&tensor, dim, 0, |acc, x| acc + x),
        _ => panic!("sum_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Mean along a dimension, keeping the dimension with size 1.
pub fn mean_dim(tensor: HostTensor, dim: usize) -> HostTensor {
    let dim_size = tensor.layout().shape()[dim];
    assert!(
        dim_size > 0,
        "mean_dim: cannot take mean of empty dimension"
    );
    let dtype = tensor.dtype();

    // Half-precision types fuse sum+divide in f32 to avoid overflow when the
    // intermediate sum exceeds f16::MAX, so they don't go through sum_dim.
    match dtype {
        DType::F16 => return mean_dim_half::<f16>(&tensor, dim),
        DType::BF16 => return mean_dim_half::<bf16>(&tensor, dim),
        _ => {}
    }

    let sum_result = sum_dim(tensor, dim);

    // Divide by dimension size
    match dtype {
        DType::F32 => scalar_div::<f32>(sum_result, dim_size as f32),
        DType::F64 => scalar_div::<f64>(sum_result, dim_size as f64),
        DType::I8 => {
            let divisor = dim_size as i32;
            let mut tensor = sum_result;
            let data: &mut [i8] = tensor.storage_mut();
            for x in data.iter_mut() {
                *x = ((*x as i32) / divisor) as i8;
            }
            tensor
        }
        DType::I16 => {
            let divisor = dim_size as i32;
            let mut tensor = sum_result;
            let data: &mut [i16] = tensor.storage_mut();
            for x in data.iter_mut() {
                *x = ((*x as i32) / divisor) as i16;
            }
            tensor
        }
        DType::I32 => scalar_div::<i32>(sum_result, dim_size as i32),
        DType::I64 => scalar_div::<i64>(sum_result, dim_size as i64),
        DType::U8 => {
            let divisor = dim_size as u32;
            let mut tensor = sum_result;
            let data: &mut [u8] = tensor.storage_mut();
            for x in data.iter_mut() {
                *x = ((*x as u32) / divisor) as u8;
            }
            tensor
        }
        DType::U16 => {
            let divisor = dim_size as u32;
            let mut tensor = sum_result;
            let data: &mut [u16] = tensor.storage_mut();
            for x in data.iter_mut() {
                *x = ((*x as u32) / divisor) as u16;
            }
            tensor
        }
        DType::U32 => scalar_div::<u32>(sum_result, dim_size as u32),
        DType::U64 => scalar_div::<u64>(sum_result, dim_size as u64),
        _ => panic!("mean_dim: unsupported dtype {:?}", dtype),
    }
}

/// Product of all elements in a tensor, returning a scalar tensor.
pub fn prod(tensor: HostTensor) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => prod_impl::<f32>(&tensor),
        DType::F64 => prod_impl::<f64>(&tensor),
        DType::F16 => reduce_scalar_half(&tensor, |a, b| a * b, 1.0, f16::to_f32, f16::from_f32),
        DType::BF16 => reduce_scalar_half(&tensor, |a, b| a * b, 1.0, bf16::to_f32, bf16::from_f32),
        DType::I8 => prod_impl_widening::<i8>(&tensor),
        DType::I16 => prod_impl_widening::<i16>(&tensor),
        DType::I32 => prod_impl_widening::<i32>(&tensor),
        DType::I64 => prod_impl::<i64>(&tensor),
        DType::U8 => prod_impl_widening::<u8>(&tensor),
        DType::U16 => prod_impl_widening::<u16>(&tensor),
        DType::U32 => prod_impl_widening::<u32>(&tensor),
        DType::U64 => prod_impl::<u64>(&tensor),
        _ => panic!("prod: unsupported dtype {:?}", tensor.dtype()),
    }
}

fn prod_impl<E: Element + bytemuck::Pod + Default + core::iter::Product>(
    tensor: &HostTensor,
) -> HostTensor {
    let result: E = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[E] = tensor.storage();
            data[start..end].iter().copied().product()
        }
        None => {
            let data: &[E] = tensor.storage();
            StridedIter::new(tensor.layout())
                .map(|idx| data[idx])
                .product()
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    HostTensor::new(
        bytes,
        Layout::contiguous(Shape::from(vec![1])),
        tensor.dtype(),
    )
}

/// Product along a dimension, keeping the dimension with size 1.
pub fn prod_dim(tensor: HostTensor, dim: usize) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => reduce_dim_f32(&tensor, dim, ReduceOp::Prod),
        DType::F64 => reduce_dim_impl::<f64, _>(&tensor, dim, 1.0, |acc, x| acc * x),
        DType::F16 => reduce_dim_half(
            &tensor,
            dim,
            1.0,
            |acc, x| acc * x,
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => reduce_dim_half(
            &tensor,
            dim,
            1.0,
            |acc, x| acc * x,
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I8 => reduce_dim_widening::<i8, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::I16 => reduce_dim_widening::<i16, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::I32 => reduce_dim_widening::<i32, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::I64 => reduce_dim_impl::<i64, _>(&tensor, dim, 1, |acc, x| acc * x),
        DType::U8 => reduce_dim_widening::<u8, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::U16 => reduce_dim_widening::<u16, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::U32 => reduce_dim_widening::<u32, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::U64 => reduce_dim_impl::<u64, _>(&tensor, dim, 1, |acc, x| acc * x),
        _ => panic!("prod_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

mod scalar_extrema;
pub use scalar_extrema::*;

// ============================================================================
// Argmax / Argmin
// ============================================================================

/// Argmax along a dimension, returning indices as isize (INDEX_DTYPE).
pub fn argmax(tensor: HostTensor, dim: usize) -> HostTensor {
    assert_dim_fits_isize(tensor.layout().shape()[dim], dim);
    // f32 last-dim fast path: 2-pass SIMD for large rows, 1-pass scalar for small rows
    if tensor.dtype() == DType::F32 && dim == tensor.layout().shape().num_dims() - 1 {
        #[cfg(feature = "simd")]
        if tensor.layout().shape()[dim] >= EXTREMUM_SIMD_ROW_THRESHOLD {
            return extremum_indices_f32_last_simd(&tensor, dim, kernels::max_f32);
        }
        return extremum_indices_f32_last_scalar(&tensor, dim, |a, b| a > b);
    }
    match tensor.dtype() {
        DType::F32 => {
            extremum_dim_with_indices::<f32, _>(&tensor, dim, |a, b| {
                !b.is_nan() && (a.is_nan() || a > b)
            })
            .1
        }
        DType::F64 => {
            extremum_dim_with_indices::<f64, _>(&tensor, dim, |a, b| {
                !b.is_nan() && (a.is_nan() || a > b)
            })
            .1
        }
        DType::F16 => {
            extremum_dim_with_indices_half::<f16, _>(
                &tensor,
                dim,
                |a, b| !b.is_nan() && (a.is_nan() || a > b),
                f16::to_f32,
                f16::from_f32,
            )
            .1
        }
        DType::BF16 => {
            extremum_dim_with_indices_half::<bf16, _>(
                &tensor,
                dim,
                |a, b| !b.is_nan() && (a.is_nan() || a > b),
                bf16::to_f32,
                bf16::from_f32,
            )
            .1
        }
        DType::I8 => extremum_dim_with_indices::<i8, _>(&tensor, dim, |a, b| a > b).1,
        DType::I16 => extremum_dim_with_indices::<i16, _>(&tensor, dim, |a, b| a > b).1,
        DType::I32 => extremum_dim_with_indices::<i32, _>(&tensor, dim, |a, b| a > b).1,
        DType::I64 => extremum_dim_with_indices::<i64, _>(&tensor, dim, |a, b| a > b).1,
        _ => panic!("argmax: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Argmin along a dimension, returning indices as isize (INDEX_DTYPE).
pub fn argmin(tensor: HostTensor, dim: usize) -> HostTensor {
    assert_dim_fits_isize(tensor.layout().shape()[dim], dim);
    // f32 last-dim fast path: 2-pass SIMD for large rows, 1-pass scalar for small rows
    if tensor.dtype() == DType::F32 && dim == tensor.layout().shape().num_dims() - 1 {
        #[cfg(feature = "simd")]
        if tensor.layout().shape()[dim] >= EXTREMUM_SIMD_ROW_THRESHOLD {
            return extremum_indices_f32_last_simd(&tensor, dim, kernels::min_f32);
        }
        return extremum_indices_f32_last_scalar(&tensor, dim, |a, b| a < b);
    }
    match tensor.dtype() {
        DType::F32 => {
            extremum_dim_with_indices::<f32, _>(&tensor, dim, |a, b| {
                !b.is_nan() && (a.is_nan() || a < b)
            })
            .1
        }
        DType::F64 => {
            extremum_dim_with_indices::<f64, _>(&tensor, dim, |a, b| {
                !b.is_nan() && (a.is_nan() || a < b)
            })
            .1
        }
        DType::F16 => {
            extremum_dim_with_indices_half::<f16, _>(
                &tensor,
                dim,
                |a, b| !b.is_nan() && (a.is_nan() || a < b),
                f16::to_f32,
                f16::from_f32,
            )
            .1
        }
        DType::BF16 => {
            extremum_dim_with_indices_half::<bf16, _>(
                &tensor,
                dim,
                |a, b| !b.is_nan() && (a.is_nan() || a < b),
                bf16::to_f32,
                bf16::from_f32,
            )
            .1
        }
        DType::I8 => extremum_dim_with_indices::<i8, _>(&tensor, dim, |a, b| a < b).1,
        DType::I16 => extremum_dim_with_indices::<i16, _>(&tensor, dim, |a, b| a < b).1,
        DType::I32 => extremum_dim_with_indices::<i32, _>(&tensor, dim, |a, b| a < b).1,
        DType::I64 => extremum_dim_with_indices::<i64, _>(&tensor, dim, |a, b| a < b).1,
        _ => panic!("argmin: unsupported dtype {:?}", tensor.dtype()),
    }
}

mod dimension;
use dimension::*;

// ============================================================================
// Mean (all elements)
// ============================================================================

/// Mean of all elements, returning a scalar tensor.
pub fn mean(tensor: HostTensor) -> HostTensor {
    let dtype = tensor.dtype();

    // Half-precision types fuse sum+divide in f32 to avoid overflow when the
    // total sum exceeds f16::MAX.
    match dtype {
        DType::F16 => return mean_scalar_half::<f16>(&tensor),
        DType::BF16 => return mean_scalar_half::<bf16>(&tensor),
        _ => {}
    }

    let n = tensor.layout().num_elements();
    let sum_result = sum(tensor);
    match dtype {
        DType::F32 => scalar_div::<f32>(sum_result, n as f32),
        DType::F64 => scalar_div::<f64>(sum_result, n as f64),
        _ => panic!("mean: unsupported dtype {:?}", dtype),
    }
}

// ============================================================================
// Max/Min along dimension (value + optional indices in a single pass)
// ============================================================================

/// Max along a dimension, returning only values.
pub fn max_dim(tensor: HostTensor, dim: usize) -> HostTensor {
    assert!(
        tensor.layout().shape()[dim] > 0,
        "max_dim: dimension {dim} has size 0"
    );
    if tensor.dtype() == DType::F32 && dim == tensor.layout().shape().num_dims() - 1 {
        #[cfg(feature = "simd")]
        if tensor.layout().shape()[dim] >= EXTREMUM_SIMD_ROW_THRESHOLD {
            return extremum_dim_f32_last_simd(&tensor, dim, kernels::max_f32);
        }
        return extremum_f32_last_scalar(&tensor, dim, |a, b| a > b);
    }
    match tensor.dtype() {
        DType::F32 => {
            extremum_dim::<f32, _>(&tensor, dim, |a, b| !b.is_nan() && (a.is_nan() || a > b))
        }
        DType::F64 => {
            extremum_dim::<f64, _>(&tensor, dim, |a, b| !b.is_nan() && (a.is_nan() || a > b))
        }
        DType::F16 => extremum_dim_half::<f16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a > b),
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => extremum_dim_half::<bf16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a > b),
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I64 => extremum_dim::<i64, _>(&tensor, dim, |a, b| a > b),
        DType::I32 => extremum_dim::<i32, _>(&tensor, dim, |a, b| a > b),
        DType::I16 => extremum_dim::<i16, _>(&tensor, dim, |a, b| a > b),
        DType::I8 => extremum_dim::<i8, _>(&tensor, dim, |a, b| a > b),
        DType::U64 => extremum_dim::<u64, _>(&tensor, dim, |a, b| a > b),
        DType::U32 => extremum_dim::<u32, _>(&tensor, dim, |a, b| a > b),
        DType::U16 => extremum_dim::<u16, _>(&tensor, dim, |a, b| a > b),
        DType::U8 => extremum_dim::<u8, _>(&tensor, dim, |a, b| a > b),
        _ => panic!("max_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Min along a dimension, returning only values.
pub fn min_dim(tensor: HostTensor, dim: usize) -> HostTensor {
    assert!(
        tensor.layout().shape()[dim] > 0,
        "min_dim: dimension {dim} has size 0"
    );
    if tensor.dtype() == DType::F32 && dim == tensor.layout().shape().num_dims() - 1 {
        #[cfg(feature = "simd")]
        if tensor.layout().shape()[dim] >= EXTREMUM_SIMD_ROW_THRESHOLD {
            return extremum_dim_f32_last_simd(&tensor, dim, kernels::min_f32);
        }
        return extremum_f32_last_scalar(&tensor, dim, |a, b| a < b);
    }
    match tensor.dtype() {
        DType::F32 => {
            extremum_dim::<f32, _>(&tensor, dim, |a, b| !b.is_nan() && (a.is_nan() || a < b))
        }
        DType::F64 => {
            extremum_dim::<f64, _>(&tensor, dim, |a, b| !b.is_nan() && (a.is_nan() || a < b))
        }
        DType::F16 => extremum_dim_half::<f16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a < b),
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => extremum_dim_half::<bf16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a < b),
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I64 => extremum_dim::<i64, _>(&tensor, dim, |a, b| a < b),
        DType::I32 => extremum_dim::<i32, _>(&tensor, dim, |a, b| a < b),
        DType::I16 => extremum_dim::<i16, _>(&tensor, dim, |a, b| a < b),
        DType::I8 => extremum_dim::<i8, _>(&tensor, dim, |a, b| a < b),
        DType::U64 => extremum_dim::<u64, _>(&tensor, dim, |a, b| a < b),
        DType::U32 => extremum_dim::<u32, _>(&tensor, dim, |a, b| a < b),
        DType::U16 => extremum_dim::<u16, _>(&tensor, dim, |a, b| a < b),
        DType::U8 => extremum_dim::<u8, _>(&tensor, dim, |a, b| a < b),
        _ => panic!("min_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Max along a dimension with indices, returning (values, indices) in a single pass.
pub fn max_dim_with_indices(tensor: HostTensor, dim: usize) -> (HostTensor, HostTensor) {
    let dim_len = tensor.layout().shape()[dim];
    assert!(
        dim_len > 0,
        "max_dim_with_indices: dimension {dim} has size 0"
    );
    assert_dim_fits_isize(dim_len, dim);
    if tensor.dtype() == DType::F32 && dim == tensor.layout().shape().num_dims() - 1 {
        #[cfg(feature = "simd")]
        if tensor.layout().shape()[dim] >= EXTREMUM_SIMD_ROW_THRESHOLD {
            return extremum_dim_with_indices_f32_last_simd(&tensor, dim, kernels::max_f32);
        }
        return extremum_with_indices_f32_last_scalar(&tensor, dim, |a, b| a > b);
    }
    match tensor.dtype() {
        DType::F32 => extremum_dim_with_indices::<f32, _>(&tensor, dim, |a, b| {
            !b.is_nan() && (a.is_nan() || a > b)
        }),
        DType::F64 => extremum_dim_with_indices::<f64, _>(&tensor, dim, |a, b| {
            !b.is_nan() && (a.is_nan() || a > b)
        }),
        DType::F16 => extremum_dim_with_indices_half::<f16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a > b),
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => extremum_dim_with_indices_half::<bf16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a > b),
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I64 => extremum_dim_with_indices::<i64, _>(&tensor, dim, |a, b| a > b),
        DType::I32 => extremum_dim_with_indices::<i32, _>(&tensor, dim, |a, b| a > b),
        DType::I16 => extremum_dim_with_indices::<i16, _>(&tensor, dim, |a, b| a > b),
        DType::I8 => extremum_dim_with_indices::<i8, _>(&tensor, dim, |a, b| a > b),
        DType::U64 => extremum_dim_with_indices::<u64, _>(&tensor, dim, |a, b| a > b),
        DType::U32 => extremum_dim_with_indices::<u32, _>(&tensor, dim, |a, b| a > b),
        DType::U16 => extremum_dim_with_indices::<u16, _>(&tensor, dim, |a, b| a > b),
        DType::U8 => extremum_dim_with_indices::<u8, _>(&tensor, dim, |a, b| a > b),
        _ => panic!(
            "max_dim_with_indices: unsupported dtype {:?}",
            tensor.dtype()
        ),
    }
}

/// Min along a dimension with indices, returning (values, indices) in a single pass.
pub fn min_dim_with_indices(tensor: HostTensor, dim: usize) -> (HostTensor, HostTensor) {
    let dim_len = tensor.layout().shape()[dim];
    assert!(
        dim_len > 0,
        "min_dim_with_indices: dimension {dim} has size 0"
    );
    assert_dim_fits_isize(dim_len, dim);
    if tensor.dtype() == DType::F32 && dim == tensor.layout().shape().num_dims() - 1 {
        #[cfg(feature = "simd")]
        if tensor.layout().shape()[dim] >= EXTREMUM_SIMD_ROW_THRESHOLD {
            return extremum_dim_with_indices_f32_last_simd(&tensor, dim, kernels::min_f32);
        }
        return extremum_with_indices_f32_last_scalar(&tensor, dim, |a, b| a < b);
    }
    match tensor.dtype() {
        DType::F32 => extremum_dim_with_indices::<f32, _>(&tensor, dim, |a, b| {
            !b.is_nan() && (a.is_nan() || a < b)
        }),
        DType::F64 => extremum_dim_with_indices::<f64, _>(&tensor, dim, |a, b| {
            !b.is_nan() && (a.is_nan() || a < b)
        }),
        DType::F16 => extremum_dim_with_indices_half::<f16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a < b),
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => extremum_dim_with_indices_half::<bf16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a < b),
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I64 => extremum_dim_with_indices::<i64, _>(&tensor, dim, |a, b| a < b),
        DType::I32 => extremum_dim_with_indices::<i32, _>(&tensor, dim, |a, b| a < b),
        DType::I16 => extremum_dim_with_indices::<i16, _>(&tensor, dim, |a, b| a < b),
        DType::I8 => extremum_dim_with_indices::<i8, _>(&tensor, dim, |a, b| a < b),
        DType::U64 => extremum_dim_with_indices::<u64, _>(&tensor, dim, |a, b| a < b),
        DType::U32 => extremum_dim_with_indices::<u32, _>(&tensor, dim, |a, b| a < b),
        DType::U16 => extremum_dim_with_indices::<u16, _>(&tensor, dim, |a, b| a < b),
        DType::U8 => extremum_dim_with_indices::<u8, _>(&tensor, dim, |a, b| a < b),
        _ => panic!(
            "min_dim_with_indices: unsupported dtype {:?}",
            tensor.dtype()
        ),
    }
}

mod extrema;
use extrema::*;

// ============================================================================
// Scalar division helpers
// ============================================================================

fn scalar_div<E: Element + bytemuck::Pod + core::ops::Div<Output = E> + Copy>(
    mut tensor: HostTensor,
    divisor: E,
) -> HostTensor {
    let data: &mut [E] = tensor.storage_mut();
    for x in data.iter_mut() {
        *x = *x / divisor;
    }
    tensor
}


pub mod dispatch;
