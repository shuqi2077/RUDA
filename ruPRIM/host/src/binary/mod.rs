//! Binary tensor operations (add, sub, mul, div).

use alloc::vec::Vec;
use ruda_core::tensor::{DType, element::Element};
use ruda_core::{bytes::Bytes, tensor::Shape};
use half::{bf16, f16};

use ruda_core::tensor::host::HostTensor;
use ruda_core::tensor::host::layout::Layout;
use ruda_core::tensor::host::strided_index::StridedIter;

#[cfg(feature = "simd")]
use crate::simd;

/// Operation type for SIMD dispatch.
#[derive(Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Apply a binary operation element-wise to two tensors.
///
/// Requires tensors to have the same shape. Uses SIMD acceleration for f32
/// when available and both tensors are contiguous.
///
/// Pass `simd_hint` to enable direct SIMD dispatch for standard ops (add/sub/mul/div).
/// Pass `None` for custom operations that have no SIMD fast path.
pub fn binary_op<F32Op, F64Op>(
    lhs: HostTensor,
    rhs: HostTensor,
    f32_op: F32Op,
    f64_op: F64Op,
    simd_hint: Option<BinaryOp>,
) -> HostTensor
where
    F32Op: Fn(f32, f32) -> f32 + Copy,
    F64Op: Fn(f64, f64) -> f64 + Copy,
{
    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "binary_op: dtype mismatch");

    // Broadcast tensors to the same shape if needed
    let (lhs, rhs) = crate::expand::broadcast_binary(lhs, rhs);

    let dtype = lhs.dtype();

    match dtype {
        DType::F32 => binary_op_f32(lhs, &rhs, f32_op, simd_hint),
        DType::F64 => binary_op_typed(lhs, &rhs, f64_op),
        DType::F16 => binary_op_typed(lhs, &rhs, |a: f16, b: f16| {
            f16::from_f32(f32_op(a.to_f32(), b.to_f32()))
        }),
        DType::BF16 => binary_op_typed(lhs, &rhs, |a: bf16, b: bf16| {
            bf16::from_f32(f32_op(a.to_f32(), b.to_f32()))
        }),
        _ => panic!("binary_op: unsupported dtype {:?}", dtype),
    }
}

#[cfg(feature = "simd")]
mod broadcast;
#[cfg(feature = "simd")]
use broadcast::*;

/// Fallback when SIMD is disabled.
#[cfg(not(feature = "simd"))]
fn binary_op_f32<Op>(
    lhs: HostTensor,
    rhs: &HostTensor,
    op: Op,
    _simd_hint: Option<BinaryOp>,
) -> HostTensor
where
    Op: Fn(f32, f32) -> f32,
{
    binary_op_typed(lhs, rhs, op)
}

/// Binary operation with in-place optimization for Pod types.
pub fn binary_op_typed<E, Op>(mut lhs: HostTensor, rhs: &HostTensor, op: Op) -> HostTensor
where
    E: Element + bytemuck::Pod,
    Op: Fn(E, E) -> E,
{
    let rhs_storage: &[E] = rhs.storage();

    // In-place fast path: lhs unique, contiguous at offset 0, rhs contiguous
    if lhs.is_unique()
        && let (Some((0, l_end)), Some((r_start, r_end))) = (
            lhs.layout().contiguous_offsets(),
            rhs.layout().contiguous_offsets(),
        )
    {
        let lhs_storage: &mut [E] = lhs.storage_mut();
        let r_slice = &rhs_storage[r_start..r_end];
        for (l, &r) in lhs_storage[..l_end].iter_mut().zip(r_slice) {
            *l = op(*l, r);
        }
        return lhs;
    }

    // Allocating path
    let shape = lhs.layout().shape().clone();
    let dtype = lhs.dtype();
    let lhs_storage: &[E] = lhs.storage();

    let result: Vec<E> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        // Both contiguous (but lhs not at offset 0)
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| op(a, b))
                .collect()
        }
        // Fast path for 2D non-contiguous (common for transpose)
        _ if lhs.layout().num_dims() == 2 => {
            apply_2d_strided(lhs_storage, rhs_storage, lhs.layout(), rhs.layout(), op)
        }
        // General fallback
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            lhs_iter
                .zip(rhs_iter)
                .map(|(li, ri)| op(lhs_storage[li], rhs_storage[ri]))
                .collect()
        }
    };

    make_tensor(result, shape, dtype)
}

/// Fast 2D strided binary operation using row-based iteration.
#[inline]
pub(crate) fn apply_2d_strided<E, R, Op>(
    lhs: &[E],
    rhs: &[E],
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> Vec<R>
where
    E: Copy,
    Op: Fn(E, E) -> R,
{
    let (rows, cols, l_row_stride, l_col_stride) = lhs_layout.as_2d_strides().unwrap();
    let (_, _, r_row_stride, r_col_stride) = rhs_layout.as_2d_strides().unwrap();
    let l_offset = lhs_layout.start_offset() as isize;
    let r_offset = rhs_layout.start_offset() as isize;

    let mut result = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        let l_row_start = l_offset + row as isize * l_row_stride;
        let r_row_start = r_offset + row as isize * r_row_stride;
        for col in 0..cols {
            let l_idx = (l_row_start + col as isize * l_col_stride) as usize;
            let r_idx = (r_row_start + col as isize * r_col_stride) as usize;
            result.push(op(lhs[l_idx], rhs[r_idx]));
        }
    }

    result
}

/// Apply a scalar operation to each element of a tensor.
///
/// Attempts in-place mutation when tensor is contiguous at offset 0.
pub fn scalar_op<F32Op, F64Op>(
    tensor: HostTensor,
    scalar: f64,
    f32_op: F32Op,
    f64_op: F64Op,
) -> HostTensor
where
    F32Op: Fn(f32, f32) -> f32 + Copy,
    F64Op: Fn(f64, f64) -> f64 + Copy,
{
    let dtype = tensor.dtype();

    match dtype {
        DType::F32 => scalar_op_typed(tensor, scalar as f32, f32_op),
        DType::F64 => scalar_op_typed(tensor, scalar, f64_op),
        DType::F16 => {
            let scalar_f16 = f16::from_f32(scalar as f32);
            let s = scalar_f16.to_f32();
            scalar_op_typed(tensor, scalar_f16, |a: f16, _| {
                f16::from_f32(f32_op(a.to_f32(), s))
            })
        }
        DType::BF16 => {
            let scalar_bf16 = bf16::from_f32(scalar as f32);
            let s = scalar_bf16.to_f32();
            scalar_op_typed(tensor, scalar_bf16, |a: bf16, _| {
                bf16::from_f32(f32_op(a.to_f32(), s))
            })
        }
        _ => panic!("scalar_op: unsupported dtype {:?}", dtype),
    }
}

pub fn scalar_op_typed<E, Op>(mut tensor: HostTensor, scalar: E, op: Op) -> HostTensor
where
    E: Element + bytemuck::Pod,
    Op: Fn(E, E) -> E,
{
    // In-place fast path: unique, contiguous at offset 0
    if tensor.is_unique()
        && let Some((0, end)) = tensor.layout().contiguous_offsets()
    {
        let storage: &mut [E] = tensor.storage_mut();
        for x in storage[..end].iter_mut() {
            *x = op(*x, scalar);
        }
        return tensor;
    }

    // Allocating path
    let shape = tensor.layout().shape().clone();
    let dtype = tensor.dtype();
    let storage: &[E] = tensor.storage();

    let result: Vec<E> = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => storage[start..end].iter().map(|&x| op(x, scalar)).collect(),
        None => StridedIter::new(tensor.layout())
            .map(|i| op(storage[i], scalar))
            .collect(),
    };

    make_tensor(result, shape, dtype)
}

/// Helper to construct a tensor from result data.
fn make_tensor<E: bytemuck::Pod + Send + Sync>(
    data: Vec<E>,
    shape: Shape,
    dtype: DType,
) -> HostTensor {
    let bytes = Bytes::from_elems(data);
    let layout = Layout::contiguous(shape);
    HostTensor::new(bytes, layout, dtype)
}

/// Apply a binary operation element-wise to two integer tensors.
///
/// Supports all integer dtypes: I64, I32, I16, I8, U64, U32, U16, U8.
pub fn int_binary_op<Op>(lhs: HostTensor, rhs: HostTensor, op: Op) -> HostTensor
where
    Op: Fn(i64, i64) -> i64 + Copy,
{
    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "int_binary_op: dtype mismatch");

    // Broadcast tensors to the same shape if needed
    let (lhs, rhs) = crate::expand::broadcast_binary(lhs, rhs);

    let dtype = lhs.dtype();

    match dtype {
        DType::I64 => binary_op_typed(lhs, &rhs, op),
        DType::I32 => binary_op_typed(lhs, &rhs, |a: i32, b: i32| op(a as i64, b as i64) as i32),
        DType::I16 => binary_op_typed(lhs, &rhs, |a: i16, b: i16| op(a as i64, b as i64) as i16),
        DType::I8 => binary_op_typed(lhs, &rhs, |a: i8, b: i8| op(a as i64, b as i64) as i8),
        // u64 values > i64::MAX wrap to negative i64. This is correct for
        // add/sub/mul/bitwise (two's complement). Div/rem are handled at the call site.
        DType::U64 => binary_op_typed(lhs, &rhs, |a: u64, b: u64| op(a as i64, b as i64) as u64),
        DType::U32 => binary_op_typed(lhs, &rhs, |a: u32, b: u32| op(a as i64, b as i64) as u32),
        DType::U16 => binary_op_typed(lhs, &rhs, |a: u16, b: u16| op(a as i64, b as i64) as u16),
        DType::U8 => binary_op_typed(lhs, &rhs, |a: u8, b: u8| op(a as i64, b as i64) as u8),
        _ => panic!("int_binary_op: unsupported dtype {:?}", dtype),
    }
}

/// Apply a scalar operation to each element of an integer tensor.
/// Note: scalar is truncated to target dtype (matches PyTorch).
pub fn int_scalar_op<Op>(tensor: HostTensor, scalar: i64, op: Op) -> HostTensor
where
    Op: Fn(i64, i64) -> i64 + Copy,
{
    let dtype = tensor.dtype();

    match dtype {
        DType::I64 => scalar_op_typed(tensor, scalar, op),
        DType::I32 => scalar_op_typed(tensor, scalar as i32, |a: i32, b: i32| {
            op(a as i64, b as i64) as i32
        }),
        DType::I16 => scalar_op_typed(tensor, scalar as i16, |a: i16, b: i16| {
            op(a as i64, b as i64) as i16
        }),
        DType::I8 => scalar_op_typed(tensor, scalar as i8, |a: i8, b: i8| {
            op(a as i64, b as i64) as i8
        }),
        DType::U64 => scalar_op_typed(tensor, scalar as u64, |a: u64, b: u64| {
            op(a as i64, b as i64) as u64
        }),
        DType::U32 => scalar_op_typed(tensor, scalar as u32, |a: u32, b: u32| {
            op(a as i64, b as i64) as u32
        }),
        DType::U16 => scalar_op_typed(tensor, scalar as u16, |a: u16, b: u16| {
            op(a as i64, b as i64) as u16
        }),
        DType::U8 => scalar_op_typed(tensor, scalar as u8, |a: u8, b: u8| {
            op(a as i64, b as i64) as u8
        }),
        _ => panic!("int_scalar_op: unsupported dtype {:?}", dtype),
    }
}

// Tests kept here exercise flex-specific behavior of `binary_op` /
// `scalar_op`: non-contiguous (transposed/narrowed/permuted) strides,
// flex f16/bf16 half-precision storage paths, and broadcast patterns
// that probe the flex layout system. Plain contiguous add/sub/mul/div
// and scalar-op smoke tests have been dropped in favor of the
// equivalent coverage in ruda-backend-tests, which exercises every
// backend. When adding new tests, keep them here only if they probe
// flex-internal dispatch; otherwise add them to
// crates/ruda-backend-tests/tests/tensor/float/ops/.
#[cfg(test)]
mod tests;

pub mod dispatch_float;
pub mod dispatch_int;
mod integer_power;
