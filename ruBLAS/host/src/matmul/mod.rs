//! Matrix multiplication via gemm crate.
//!
//! Optimizations:
//! - Strided gemm for f32/f64/f16 avoids copying non-contiguous tensors
//! - Enables parallelism for large matrices (with rayon feature)
//! - Batched matmul parallelized across batch dimension

use alloc::vec;
use alloc::vec::Vec;
use ruda_core::tensor::{DType, element::Element};
use ruda_core::{bytes::Bytes, tensor::Shape};
use half::{bf16, f16};

use ruda_core::tensor::host::{HostTensor, Layout};

/// Types that can be used with gemm-based matmul.
/// Only implement for types that `gemm::gemm` dispatches on via TypeId (f32, f64, f16).
trait GemmScalar: Element + bytemuck::Pod {
    fn zero() -> Self;
    fn one() -> Self;
}

impl GemmScalar for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl GemmScalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl GemmScalar for f16 {
    fn zero() -> Self {
        f16::from_f32(0.0)
    }
    fn one() -> Self {
        f16::from_f32(1.0)
    }
}

/// Checked multiplication for matrix sizes, panics on overflow.
#[inline]
fn checked_size(a: usize, b: usize) -> usize {
    a.checked_mul(b)
        .unwrap_or_else(|| panic!("matmul: matrix size overflow: {a} * {b}"))
}

/// Threshold for enabling parallelism (M*N*K operations).
/// 192^3 = ~7M ops - balance between 128x128 (no parallel) and 256x256 (parallel)
const PARALLEL_THRESHOLD: usize = 192 * 192 * 192;

/// Threshold for batch-level parallelism (total ops across all batches).
/// Use batch parallelism when individual matrices are small but total work is large.
#[cfg(feature = "rayon")]
const BATCH_PARALLEL_THRESHOLD: usize = 128 * 128 * 128; // ~2M ops total

/// Get parallelism setting based on matrix size.
fn get_parallelism(m: usize, n: usize, k: usize) -> gemm::Parallelism {
    let ops = m.saturating_mul(n).saturating_mul(k);
    if ops >= PARALLEL_THRESHOLD {
        #[cfg(feature = "rayon")]
        {
            gemm::Parallelism::Rayon(0) // 0 = use all available threads
        }
        #[cfg(not(feature = "rayon"))]
        {
            gemm::Parallelism::None
        }
    } else {
        gemm::Parallelism::None
    }
}

/// Dispatch matrix multiplication based on dtype.
pub fn matmul(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    assert_eq!(lhs.dtype(), rhs.dtype(), "matmul: dtype mismatch");

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    assert!(lhs_rank >= 2, "matmul requires at least 2D tensors");
    assert!(rhs_rank >= 2, "matmul requires at least 2D tensors");

    // Check inner dimensions match: lhs[..., M, K] x rhs[..., K, N]
    let k_lhs = lhs_shape[lhs_rank - 1];
    let k_rhs = rhs_shape[rhs_rank - 2];
    assert_eq!(k_lhs, k_rhs, "matmul: inner dimensions must match");

    match lhs.dtype() {
        DType::F32 => matmul_gemm::<f32>(lhs, rhs),
        DType::F64 => matmul_gemm::<f64>(lhs, rhs),
        DType::F16 => matmul_gemm::<f16>(lhs, rhs),
        DType::BF16 => matmul_bf16(lhs, rhs),
        _ => panic!("matmul: unsupported dtype {:?}", lhs.dtype()),
    }
}

mod shape;
use shape::*;

// ============================================================================
// Generic gemm-based matmul (f32, f64, f16)
// ============================================================================

fn matmul_gemm<T: GemmScalar>(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    let lhs_rank = lhs.layout().shape().num_dims();
    let rhs_rank = rhs.layout().shape().num_dims();

    if lhs_rank == 2 && rhs_rank == 2 {
        matmul_2d_strided::<T>(lhs, rhs)
    } else {
        matmul_batched_gemm::<T>(lhs, rhs)
    }
}

/// 2D matmul with strided support: [M, K] x [K, N] -> [M, N]
fn matmul_2d_strided<T: GemmScalar>(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();

    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let n = rhs_shape[1];

    let (lhs_row_stride, lhs_col_stride) = get_2d_strides(lhs.layout());
    let (rhs_row_stride, rhs_col_stride) = get_2d_strides(rhs.layout());

    let lhs_data: &[T] = lhs.storage();
    let rhs_data: &[T] = rhs.storage();
    let lhs_ptr = unsafe { lhs_data.as_ptr().add(lhs.layout().start_offset()) };
    let rhs_ptr = unsafe { rhs_data.as_ptr().add(rhs.layout().start_offset()) };

    let out_shape = Shape::from(vec![m, n]);
    let mut output = HostTensor::empty(out_shape, T::dtype());
    let out_data: &mut [T] = output.storage_mut();

    let parallelism = get_parallelism(m, n, k);

    unsafe {
        gemm_call(
            m,
            n,
            k,
            out_data.as_mut_ptr(),
            1,
            n as isize,
            lhs_ptr,
            lhs_col_stride,
            lhs_row_stride,
            rhs_ptr,
            rhs_col_stride,
            rhs_row_stride,
            parallelism,
        );
    }

    output
}

/// Strided gemm call for one matrix. Wraps `gemm::gemm` with GemmScalar zero/one.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_call<T: GemmScalar>(
    m: usize,
    n: usize,
    k: usize,
    out: *mut T,
    out_cs: isize,
    out_rs: isize,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    parallelism: gemm::Parallelism,
) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out,
            out_cs,
            out_rs,
            false,
            lhs,
            lhs_cs,
            lhs_rs,
            rhs,
            rhs_cs,
            rhs_rs,
            T::zero(),
            T::one(),
            false,
            false,
            false,
            parallelism,
        );
    }
}

/// Batched matmul: [B..., M, K] x [B..., K, N] -> [B..., M, N]
/// Supports broadcasting on batch dimensions and strided (non-contiguous) inputs.
fn matmul_batched_gemm<T: GemmScalar>(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    let m = lhs_shape[lhs_rank - 2];
    let k = lhs_shape[lhs_rank - 1];
    let n = rhs_shape[rhs_rank - 1];

    let lhs_batch: Vec<usize> = lhs_shape[..lhs_rank - 2].to_vec();
    let rhs_batch: Vec<usize> = rhs_shape[..rhs_rank - 2].to_vec();

    let (broadcast_shape, _, _) = broadcast_batch_dims(&lhs_batch, &rhs_batch);
    let batch_size: usize = broadcast_shape.iter().product();
    let broadcast_len = broadcast_shape.len();

    let lhs_batch_strides =
        broadcast_batch_elem_strides(&lhs_batch, lhs.layout().strides(), broadcast_len);
    let rhs_batch_strides =
        broadcast_batch_elem_strides(&rhs_batch, rhs.layout().strides(), broadcast_len);

    let (lhs_row_stride, lhs_col_stride) = get_2d_strides(lhs.layout());
    let (rhs_row_stride, rhs_col_stride) = get_2d_strides(rhs.layout());

    let out_matrix_size = checked_size(m, n);

    let mut out_dims = broadcast_shape.clone();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);

    let mut output = HostTensor::empty(out_shape, T::dtype());

    let lhs_data: &[T] = lhs.storage();
    let rhs_data: &[T] = rhs.storage();
    let lhs_start = lhs.layout().start_offset() as isize;
    let rhs_start = rhs.layout().start_offset() as isize;
    let out_data: &mut [T] = output.storage_mut();

    let per_matrix_ops = m.saturating_mul(n).saturating_mul(k);

    // Closure: run gemm for one batch slice at the given pointers
    let run_one = |out_ptr: *mut T, b: usize, parallelism: gemm::Parallelism| {
        let lhs_off = lhs_start + batch_elem_offset(b, &broadcast_shape, &lhs_batch_strides);
        let rhs_off = rhs_start + batch_elem_offset(b, &broadcast_shape, &rhs_batch_strides);
        unsafe {
            gemm_call::<T>(
                m,
                n,
                k,
                out_ptr,
                1,
                n as isize,
                lhs_data.as_ptr().offset(lhs_off),
                lhs_col_stride,
                lhs_row_stride,
                rhs_data.as_ptr().offset(rhs_off),
                rhs_col_stride,
                rhs_row_stride,
                parallelism,
            );
        }
    };

    // Strategy:
    // 1. Large matrices: let gemm parallelize internally
    // 2. Small matrices, large batch: parallelize batch loop
    // 3. Small total work: single-threaded
    #[cfg(feature = "rayon")]
    {
        let total_ops = batch_size.saturating_mul(per_matrix_ops);
        let prefer_batch_parallel = batch_size >= 4 && total_ops >= BATCH_PARALLEL_THRESHOLD;

        if per_matrix_ops >= PARALLEL_THRESHOLD && !prefer_batch_parallel {
            let parallelism = gemm::Parallelism::Rayon(0);
            for b in 0..batch_size {
                run_one(out_data[b * out_matrix_size..].as_mut_ptr(), b, parallelism);
            }
        } else if total_ops >= BATCH_PARALLEL_THRESHOLD && batch_size > 1 {
            use rayon::prelude::*;

            out_data
                .par_chunks_mut(out_matrix_size)
                .enumerate()
                .for_each(|(b, out_chunk)| {
                    run_one(out_chunk.as_mut_ptr(), b, gemm::Parallelism::None);
                });
        } else {
            for b in 0..batch_size {
                run_one(
                    out_data[b * out_matrix_size..].as_mut_ptr(),
                    b,
                    gemm::Parallelism::None,
                );
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        let _ = per_matrix_ops;
        for b in 0..batch_size {
            run_one(
                out_data[b * out_matrix_size..].as_mut_ptr(),
                b,
                gemm::Parallelism::None,
            );
        }
    }

    output
}

// ============================================================================
// bf16 matmul (via f32 conversion)
// ============================================================================

fn matmul_bf16(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    let lhs = lhs.to_contiguous();
    let rhs = rhs.to_contiguous();

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();

    // Convert bf16 -> f32
    let lhs_f32: Vec<f32> = lhs.storage::<bf16>().iter().map(|x| x.to_f32()).collect();
    let rhs_f32: Vec<f32> = rhs.storage::<bf16>().iter().map(|x| x.to_f32()).collect();

    // Create f32 tensors
    let lhs_f32_tensor = HostTensor::new(
        Bytes::from_elems(lhs_f32),
        Layout::contiguous(lhs_shape.clone()),
        DType::F32,
    );
    let rhs_f32_tensor = HostTensor::new(
        Bytes::from_elems(rhs_f32),
        Layout::contiguous(rhs_shape.clone()),
        DType::F32,
    );

    // Compute matmul in f32
    let result_f32 = matmul_gemm::<f32>(lhs_f32_tensor, rhs_f32_tensor);

    // Convert f32 -> bf16
    let result_bf16: Vec<bf16> = result_f32
        .storage::<f32>()
        .iter()
        .map(|x| bf16::from_f32(*x))
        .collect();

    HostTensor::new(
        Bytes::from_elems(result_bf16),
        result_f32.layout().clone(),
        DType::BF16,
    )
}

mod integer;
pub use integer::int_matmul;

#[cfg(test)]
mod tests;
