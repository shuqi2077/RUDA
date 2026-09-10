//! Comparison operations returning boolean tensors.

use alloc::boxed::Box;
#[cfg(feature = "simd")]
use alloc::vec;
use alloc::vec::Vec;
use ruda_core::tensor::{DType, element::Element};
use ruda_core::{bytes::Bytes, tensor::{BoolDType, BoolStore, Shape}};
use half::{bf16, f16};
use bytemuck::Pod;

use ruda_core::tensor::host::strided_index::StridedIter;
use ruda_core::tensor::host::{HostTensor, Layout};

use crate::simd;

/// Comparison operation type for SIMD dispatch.
pub use simd::CmpOp as CompareOp;

/// Compare two tensors element-wise, returning a boolean tensor with the
/// requested output dtype.
pub fn compare<F32Cmp, F64Cmp>(
    lhs: HostTensor,
    rhs: HostTensor,
    out_dtype: BoolDType,
    f32_cmp: F32Cmp,
    f64_cmp: F64Cmp,
    simd_hint: Option<CompareOp>,
) -> HostTensor
where
    F32Cmp: Fn(f32, f32) -> bool + Copy,
    F64Cmp: Fn(f64, f64) -> bool + Copy,
{
    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "compare: dtype mismatch");

    // Broadcast to same shape if needed
    let (lhs, rhs) = crate::expand::broadcast_binary(lhs, rhs);

    let dtype = lhs.dtype();

    match dtype {
        DType::F32 => compare_f32(lhs, &rhs, out_dtype, f32_cmp, simd_hint),
        DType::F64 => compare_typed(lhs, &rhs, out_dtype, f64_cmp),
        DType::F16 => compare_typed(lhs, &rhs, out_dtype, |a: f16, b: f16| {
            f32_cmp(a.to_f32(), b.to_f32())
        }),
        DType::BF16 => compare_typed(lhs, &rhs, out_dtype, |a: bf16, b: bf16| {
            f32_cmp(a.to_f32(), b.to_f32())
        }),
        _ => panic!("compare: unsupported dtype {:?}", dtype),
    }
}

/// Specialized comparison for f32 with SIMD fast path.
#[cfg(feature = "simd")]
fn compare_f32<Cmp>(
    lhs: HostTensor,
    rhs: &HostTensor,
    out_dtype: BoolDType,
    cmp: Cmp,
    simd_hint: Option<CompareOp>,
) -> HostTensor
where
    Cmp: Fn(f32, f32) -> bool,
{
    // SIMD fast path: both tensors contiguous
    if let (Some((l_start, l_end)), Some((r_start, r_end))) = (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) && let Some(simd_op) = simd_hint
    {
        let shape = lhs.layout().shape().clone();
        let lhs_storage: &[f32] = lhs.storage();
        let rhs_storage: &[f32] = rhs.storage();

        let l_slice = &lhs_storage[l_start..l_end];
        let r_slice = &rhs_storage[r_start..r_end];

        let mut result = vec![0u8; l_slice.len()];
        simd::cmp_f32(l_slice, r_slice, &mut result, simd_op);

        return make_bool_tensor(result, shape, out_dtype);
    }

    // Optimized broadcast path for outer-product style broadcasting
    // Pattern: [N, 1] vs [1, M] -> [N, M] where one has stride 0 in inner dim
    if lhs.layout().num_dims() == 2
        && let Some(simd_op) = simd_hint
        && let Some((result, shape)) = try_broadcast_cmp_f32(&lhs, rhs, simd_op)
    {
        return make_bool_tensor(result, shape, out_dtype);
    }

    // Fallback to generic path
    compare_typed(lhs, rhs, out_dtype, cmp)
}

/// Try optimized outer-product style broadcast comparison.
/// Returns Some((result, shape)) if the pattern matches.
#[cfg(feature = "simd")]
fn try_broadcast_cmp_f32(
    lhs: &HostTensor,
    rhs: &HostTensor,
    op: simd::CmpOp,
) -> Option<(Vec<u8>, Shape)> {
    let lhs_strides = lhs.layout().strides();
    let rhs_strides = rhs.layout().strides();
    let shape = lhs.layout().shape().clone();
    let [rows, cols] = shape[..] else {
        return None;
    };

    // Pattern 1: lhs has stride 0 in dim 1 (column broadcast), rhs contiguous
    // lhs[i,j] = lhs_data[i*stride], rhs[i,j] = rhs_data[i*cols + j]
    if lhs_strides[1] == 0 && rhs_strides == [cols as isize, 1] {
        let lhs_storage: &[f32] = lhs.storage();
        let rhs_storage: &[f32] = rhs.storage();
        let l_offset = lhs.layout().start_offset() as isize;
        let l_stride = lhs_strides[0];
        let r_offset = rhs.layout().start_offset();

        let mut result = vec![0u8; rows * cols];
        for row in 0..rows {
            let a_val = lhs_storage[(l_offset + row as isize * l_stride) as usize];
            let r_row_start = r_offset + row * cols;
            let r_slice = &rhs_storage[r_row_start..r_row_start + cols];
            let out_start = row * cols;
            simd::cmp_scalar_f32(
                r_slice,
                a_val,
                &mut result[out_start..out_start + cols],
                swap_cmp_op(op),
            );
        }
        return Some((result, shape));
    }

    // Pattern 2: rhs has stride 0 in dim 0 (row broadcast), lhs contiguous
    // lhs[i,j] = lhs_data[i*cols + j], rhs[i,j] = rhs_data[j*stride]
    if rhs_strides[0] == 0 && lhs_strides == [cols as isize, 1] {
        let lhs_storage: &[f32] = lhs.storage();
        let rhs_storage: &[f32] = rhs.storage();
        let l_offset = lhs.layout().start_offset();
        let r_offset = rhs.layout().start_offset() as isize;
        let r_stride = rhs_strides[1];

        // Build the broadcast rhs values once
        let rhs_row: Vec<f32> = (0..cols)
            .map(|j| rhs_storage[(r_offset + j as isize * r_stride) as usize])
            .collect();

        let mut result = vec![0u8; rows * cols];
        for row in 0..rows {
            let l_row_start = l_offset + row * cols;
            let l_slice = &lhs_storage[l_row_start..l_row_start + cols];
            let out_start = row * cols;
            // Compare row with broadcast values
            for (j, (&lv, &rv)) in l_slice.iter().zip(rhs_row.iter()).enumerate() {
                result[out_start + j] = match op {
                    simd::CmpOp::Gt => (lv > rv) as u8,
                    simd::CmpOp::Ge => (lv >= rv) as u8,
                    simd::CmpOp::Lt => (lv < rv) as u8,
                    simd::CmpOp::Le => (lv <= rv) as u8,
                    simd::CmpOp::Eq => (lv == rv) as u8,
                    simd::CmpOp::Ne => (lv != rv) as u8,
                };
            }
        }
        return Some((result, shape));
    }

    // Pattern 3: Outer product - lhs stride 0 in dim 1, rhs stride 0 in dim 0
    // This is the [N,1] vs [1,M] case
    if lhs_strides[1] == 0 && rhs_strides[0] == 0 {
        let lhs_storage: &[f32] = lhs.storage();
        let rhs_storage: &[f32] = rhs.storage();
        let l_offset = lhs.layout().start_offset() as isize;
        let l_stride = lhs_strides[0];
        let r_offset = rhs.layout().start_offset() as isize;
        let r_stride = rhs_strides[1];

        // Build the broadcast rhs row once
        let rhs_row: Vec<f32> = (0..cols)
            .map(|j| rhs_storage[(r_offset + j as isize * r_stride) as usize])
            .collect();

        let mut result = vec![0u8; rows * cols];
        for row in 0..rows {
            let a_val = lhs_storage[(l_offset + row as isize * l_stride) as usize];
            let out_start = row * cols;
            simd::cmp_scalar_f32(
                &rhs_row,
                a_val,
                &mut result[out_start..out_start + cols],
                swap_cmp_op(op),
            );
        }
        return Some((result, shape));
    }

    None
}

/// Swap comparison operation for reversed operand order.
#[cfg(feature = "simd")]
fn swap_cmp_op(op: simd::CmpOp) -> simd::CmpOp {
    match op {
        simd::CmpOp::Gt => simd::CmpOp::Lt, // a > b becomes b < a
        simd::CmpOp::Ge => simd::CmpOp::Le,
        simd::CmpOp::Lt => simd::CmpOp::Gt,
        simd::CmpOp::Le => simd::CmpOp::Ge,
        simd::CmpOp::Eq => simd::CmpOp::Eq, // symmetric
        simd::CmpOp::Ne => simd::CmpOp::Ne,
    }
}

/// Fallback when SIMD is disabled.
#[cfg(not(feature = "simd"))]
fn compare_f32<Cmp>(
    lhs: HostTensor,
    rhs: &HostTensor,
    out_dtype: BoolDType,
    cmp: Cmp,
    _simd_hint: Option<CompareOp>,
) -> HostTensor
where
    Cmp: Fn(f32, f32) -> bool,
{
    compare_typed(lhs, rhs, out_dtype, cmp)
}

/// Compare tensor with scalar, returning a boolean tensor with the requested
/// output dtype.
pub fn compare_elem<F32Cmp, F64Cmp>(
    lhs: HostTensor,
    rhs: f64,
    out_dtype: BoolDType,
    f32_cmp: F32Cmp,
    f64_cmp: F64Cmp,
    simd_hint: Option<CompareOp>,
) -> HostTensor
where
    F32Cmp: Fn(f32, f32) -> bool + Copy,
    F64Cmp: Fn(f64, f64) -> bool + Copy,
{
    let dtype = lhs.dtype();

    match dtype {
        DType::F32 => compare_elem_f32(lhs, rhs as f32, out_dtype, f32_cmp, simd_hint),
        DType::F64 => compare_elem_typed(lhs, rhs, out_dtype, f64_cmp),
        DType::F16 => {
            let scalar = f16::from_f64(rhs);
            compare_elem_typed(lhs, scalar, out_dtype, |a: f16, b: f16| {
                f32_cmp(a.to_f32(), b.to_f32())
            })
        }
        DType::BF16 => {
            let scalar = bf16::from_f64(rhs);
            compare_elem_typed(lhs, scalar, out_dtype, |a: bf16, b: bf16| {
                f32_cmp(a.to_f32(), b.to_f32())
            })
        }
        _ => panic!("compare_elem: unsupported dtype {:?}", dtype),
    }
}

/// Specialized scalar comparison for f32 with SIMD fast path.
#[cfg(feature = "simd")]
fn compare_elem_f32<Cmp>(
    lhs: HostTensor,
    rhs: f32,
    out_dtype: BoolDType,
    cmp: Cmp,
    simd_hint: Option<CompareOp>,
) -> HostTensor
where
    Cmp: Fn(f32, f32) -> bool,
{
    // SIMD fast path: tensor is contiguous
    if let Some((start, end)) = lhs.layout().contiguous_offsets()
        && let Some(simd_op) = simd_hint
    {
        let shape = lhs.layout().shape().clone();
        let lhs_storage: &[f32] = lhs.storage();
        let l_slice = &lhs_storage[start..end];

        let mut result = vec![0u8; l_slice.len()];
        simd::cmp_scalar_f32(l_slice, rhs, &mut result, simd_op);

        return make_bool_tensor(result, shape, out_dtype);
    }

    // Fallback to generic path
    compare_elem_typed(lhs, rhs, out_dtype, cmp)
}

/// Fallback when SIMD is disabled.
#[cfg(not(feature = "simd"))]
fn compare_elem_f32<Cmp>(
    lhs: HostTensor,
    rhs: f32,
    out_dtype: BoolDType,
    cmp: Cmp,
    _simd_hint: Option<CompareOp>,
) -> HostTensor
where
    Cmp: Fn(f32, f32) -> bool,
{
    compare_elem_typed(lhs, rhs, out_dtype, cmp)
}

fn compare_typed<E, Cmp>(
    lhs: HostTensor,
    rhs: &HostTensor,
    out_dtype: BoolDType,
    cmp: Cmp,
) -> HostTensor
where
    E: Element + Pod,
    Cmp: Fn(E, E) -> bool,
{
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[E] = lhs.storage();
    let rhs_storage: &[E] = rhs.storage();

    let result: Vec<u8> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| cmp(a, b) as u8)
                .collect()
        }
        // Fast path for 2D non-contiguous (common for transpose)
        _ if lhs.layout().num_dims() == 2 => crate::binary::apply_2d_strided(
            lhs_storage,
            rhs_storage,
            lhs.layout(),
            rhs.layout(),
            |a, b| cmp(a, b) as u8,
        ),
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            lhs_iter
                .zip(rhs_iter)
                .map(|(li, ri)| cmp(lhs_storage[li], rhs_storage[ri]) as u8)
                .collect()
        }
    };

    make_bool_tensor(result, shape, out_dtype)
}

fn compare_elem_typed<E, Cmp>(lhs: HostTensor, rhs: E, out_dtype: BoolDType, cmp: Cmp) -> HostTensor
where
    E: Element + Pod + Copy,
    Cmp: Fn(E, E) -> bool,
{
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[E] = lhs.storage();

    let result: Vec<u8> = match lhs.layout().contiguous_offsets() {
        Some((start, end)) => lhs_storage[start..end]
            .iter()
            .map(|&a| cmp(a, rhs) as u8)
            .collect(),
        None => StridedIter::new(lhs.layout())
            .map(|idx| cmp(lhs_storage[idx], rhs) as u8)
            .collect(),
    };

    make_bool_tensor(result, shape, out_dtype)
}

/// Build a bool `FlexTensor` from a `Vec<u8>` of 0/1 bytes, tagged with the
/// requested output dtype.
///
/// ruda-tensor-host stores bools as 1 byte per element, so only Native and U8 are
/// supported. `Bool(U32)` would require 4-byte-per-element storage throughout
/// the backend; `dtype_usage` declares it unsupported and this function panics
/// if it's requested.
pub fn make_bool_tensor(data: Vec<u8>, shape: Shape, out_dtype: BoolDType) -> HostTensor {
    let store = match out_dtype {
        BoolDType::Native => BoolStore::Native,
        BoolDType::U8 => BoolStore::U8,
        BoolDType::U32 => panic!(
            "ruda-tensor-host does not support Bool(U32) storage (only Native and U8). \
             Use a backend that declares Bool(U32) support, or work with Bool(Native)/Bool(U8)."
        ),
    };
    let bytes = Bytes::from_elems(data);
    HostTensor::new(bytes, Layout::contiguous(shape), DType::Bool(store))
}

// Specific comparison functions

pub fn greater(lhs: HostTensor, rhs: HostTensor, out_dtype: BoolDType) -> HostTensor {
    compare(
        lhs,
        rhs,
        out_dtype,
        |a, b| a > b,
        |a, b| a > b,
        Some(CompareOp::Gt),
    )
}

pub fn greater_elem(lhs: HostTensor, rhs: f64, out_dtype: BoolDType) -> HostTensor {
    compare_elem(
        lhs,
        rhs,
        out_dtype,
        |a, b| a > b,
        |a, b| a > b,
        Some(CompareOp::Gt),
    )
}

pub fn greater_equal(lhs: HostTensor, rhs: HostTensor, out_dtype: BoolDType) -> HostTensor {
    compare(
        lhs,
        rhs,
        out_dtype,
        |a, b| a >= b,
        |a, b| a >= b,
        Some(CompareOp::Ge),
    )
}

pub fn greater_equal_elem(lhs: HostTensor, rhs: f64, out_dtype: BoolDType) -> HostTensor {
    compare_elem(
        lhs,
        rhs,
        out_dtype,
        |a, b| a >= b,
        |a, b| a >= b,
        Some(CompareOp::Ge),
    )
}

pub fn lower(lhs: HostTensor, rhs: HostTensor, out_dtype: BoolDType) -> HostTensor {
    compare(
        lhs,
        rhs,
        out_dtype,
        |a, b| a < b,
        |a, b| a < b,
        Some(CompareOp::Lt),
    )
}

pub fn lower_elem(lhs: HostTensor, rhs: f64, out_dtype: BoolDType) -> HostTensor {
    compare_elem(
        lhs,
        rhs,
        out_dtype,
        |a, b| a < b,
        |a, b| a < b,
        Some(CompareOp::Lt),
    )
}

pub fn lower_equal(lhs: HostTensor, rhs: HostTensor, out_dtype: BoolDType) -> HostTensor {
    compare(
        lhs,
        rhs,
        out_dtype,
        |a, b| a <= b,
        |a, b| a <= b,
        Some(CompareOp::Le),
    )
}

pub fn lower_equal_elem(lhs: HostTensor, rhs: f64, out_dtype: BoolDType) -> HostTensor {
    compare_elem(
        lhs,
        rhs,
        out_dtype,
        |a, b| a <= b,
        |a, b| a <= b,
        Some(CompareOp::Le),
    )
}

pub fn equal(lhs: HostTensor, rhs: HostTensor, out_dtype: BoolDType) -> HostTensor {
    compare(
        lhs,
        rhs,
        out_dtype,
        |a, b| a == b,
        |a, b| a == b,
        Some(CompareOp::Eq),
    )
}

pub fn equal_elem(lhs: HostTensor, rhs: f64, out_dtype: BoolDType) -> HostTensor {
    compare_elem(
        lhs,
        rhs,
        out_dtype,
        |a, b| a == b,
        |a, b| a == b,
        Some(CompareOp::Eq),
    )
}

pub fn not_equal(lhs: HostTensor, rhs: HostTensor, out_dtype: BoolDType) -> HostTensor {
    compare(
        lhs,
        rhs,
        out_dtype,
        |a, b| a != b,
        |a, b| a != b,
        Some(CompareOp::Ne),
    )
}

pub fn not_equal_elem(lhs: HostTensor, rhs: f64, out_dtype: BoolDType) -> HostTensor {
    compare_elem(
        lhs,
        rhs,
        out_dtype,
        |a, b| a != b,
        |a, b| a != b,
        Some(CompareOp::Ne),
    )
}

mod integer;
pub use integer::*;

mod predicate_reduce;
pub use predicate_reduce::*;

// ============================================================================
// Helpers for any/all
// ============================================================================

fn bool_scalar(val: bool, out_dtype: BoolDType) -> HostTensor {
    let byte: u8 = if val { 1 } else { 0 };
    make_bool_tensor(alloc::vec![byte], Shape::from(alloc::vec![1]), out_dtype)
}

fn iter_elements<'a, E: Element + Pod + 'a>(
    tensor: &'a HostTensor,
) -> Box<dyn Iterator<Item = E> + 'a> {
    let data: &[E] = tensor.storage();
    match tensor.layout().contiguous_offsets() {
        Some((start, end)) => Box::new(data[start..end].iter().copied()),
        None => Box::new(StridedIter::new(tensor.layout()).map(move |idx| data[idx])),
    }
}

/// Reduce along a dimension producing a bool tensor.
///
/// The `is_nonzero` closure reads the data slice at a given index and returns
/// whether the element is nonzero.
fn reduce_bool_dim_with(
    tensor: &HostTensor,
    dim: usize,
    init: bool,
    combine: fn(bool, bool) -> bool,
    out_dtype: BoolDType,
    is_nonzero: impl Fn(usize) -> bool,
) -> HostTensor {
    debug_assert!(tensor.is_contiguous() && tensor.layout().start_offset() == 0);
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let out_size = outer_size.max(1) * inner_size.max(1);
    let mut result: Vec<u8> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut acc = init;
            for d in 0..dim_size {
                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                acc = combine(acc, is_nonzero(idx));
            }
            result.push(if acc { 1 } else { 0 });
        }
    }

    make_bool_tensor(result, Shape::from(out_shape), out_dtype)
}

/// Reduce along a dimension producing a bool tensor (for float any/all_dim).
fn reduce_bool_dim(
    tensor: &HostTensor,
    dim: usize,
    init: bool,
    combine: fn(bool, bool) -> bool,
    out_dtype: BoolDType,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    match tensor.dtype() {
        DType::F32 => {
            let data: &[f32] = tensor.storage();
            reduce_bool_dim_with(&tensor, dim, init, combine, out_dtype, |idx| {
                data[idx] != 0.0
            })
        }
        DType::F64 => {
            let data: &[f64] = tensor.storage();
            reduce_bool_dim_with(&tensor, dim, init, combine, out_dtype, |idx| {
                data[idx] != 0.0
            })
        }
        DType::F16 => {
            let data: &[f16] = tensor.storage();
            reduce_bool_dim_with(&tensor, dim, init, combine, out_dtype, |idx| {
                data[idx].to_f32() != 0.0
            })
        }
        DType::BF16 => {
            let data: &[bf16] = tensor.storage();
            reduce_bool_dim_with(&tensor, dim, init, combine, out_dtype, |idx| {
                data[idx].to_f32() != 0.0
            })
        }
        _ => panic!("reduce_bool_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Reduce along a dimension producing a bool tensor (for int any/all_dim).
fn reduce_bool_dim_int(
    tensor: &HostTensor,
    dim: usize,
    init: bool,
    combine: fn(bool, bool) -> bool,
    out_dtype: BoolDType,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    macro_rules! dispatch {
        ($ty:ty) => {{
            let data: &[$ty] = tensor.storage();
            reduce_bool_dim_with(&tensor, dim, init, combine, out_dtype, |idx| data[idx] != 0)
        }};
    }
    match tensor.dtype() {
        DType::I64 => dispatch!(i64),
        DType::I32 => dispatch!(i32),
        DType::I16 => dispatch!(i16),
        DType::I8 => dispatch!(i8),
        DType::U64 => dispatch!(u64),
        DType::U32 => dispatch!(u32),
        DType::U16 => dispatch!(u16),
        DType::U8 => dispatch!(u8),
        other => panic!("reduce_bool_dim_int: unsupported dtype {:?}", other),
    }
}

/// Reduce along a dimension producing a bool tensor (for bool any/all_dim).
fn reduce_bool_dim_raw(
    tensor: &HostTensor,
    dim: usize,
    init: bool,
    combine: fn(bool, bool) -> bool,
    out_dtype: BoolDType,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let data: &[u8] = tensor.bytes();
    reduce_bool_dim_with(&tensor, dim, init, combine, out_dtype, |idx| data[idx] != 0)
}

// Tests kept here probe flex-internal `reduce_bool_dim_with` dispatch on
// non-contiguous inputs (stale-pointer-read regression, see prior incident
// in `any_float_dim`). Plain comparison ops and stride variants (flipped
// / transposed / narrowed) have been migrated to ruda-backend-tests at
// tensor/{float,int}/ops/comparison.rs so every backend is exercised.
#[cfg(test)]
mod tests;

pub mod dispatch;
