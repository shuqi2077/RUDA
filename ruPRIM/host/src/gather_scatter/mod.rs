//! Gather and scatter operations for indexed tensor access.

use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;
use ruda_core::tensor::{DType, element::Element};
use ruda_core::{bytes::Bytes, tensor::Shape};
use bytemuck::Pod;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use ruda_core::tensor::host::{HostTensor, Layout};

/// Read indices from a tensor as `isize`, the native offset type used by the
/// gather/scatter/select kernels in this module.
///
/// This is the internal index layer for ruda-tensor-host: every indexed op
/// ([`gather`], [`scatter_add`], [`select`], [`select_add`], and the
/// [`scatter_min`]/[`scatter_max`] variants) routes its index tensor through
/// this helper before touching the element buffer. Normalising to `isize`
/// lets the kernels use a single inner-loop signature regardless of how the
/// caller's index tensor was dtyped.
///
/// # Accepted widths
///
/// Any of the integer DTypes `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`,
/// `U64` is accepted. This is intentional: ruda-tensor-host's default `IntElem` is
/// I32 rather than the I64 convention used by other backends, and users can
/// also pin index tensors to any width they want via
/// `Tensor::from_data(.., (&device, DType::Ix))`. Whichever width lands here
/// is converted to `isize` on the fly.
///
/// # Zero-copy vs. owned
///
/// The return type is `Cow<'_, [isize]>` because only one width is zero-copy:
/// the one matching the host pointer width. On 64-bit targets, I64 indices
/// can be borrowed directly via `bytemuck::cast_slice` (both are 8 bytes).
/// Every other width requires an owned `Vec<isize>` with an element-wise
/// cast. U64 indices additionally go through a `try_from` to surface values
/// that would wrap when cast to `isize`.
///
/// # History
///
/// Earlier versions of `int_gather`, `int_scatter_add`, `int_select`, and
/// `int_select_add` carried a `debug_assert_eq!(indices.dtype(), DType::I64,
/// ..)` that contradicted this helper's contract. The asserts were dropped
/// in tracel-ai/ruda#4776 once it was confirmed that `read_indices` had
/// always handled every supported width correctly at runtime. If you're
/// tempted to re-add a dtype check here, don't - the float siblings
/// ([`gather_f32`], [`select_f32`], ...) already share this helper without a
/// check, and asymmetry between the int and float paths was what surfaced
/// the bug.
fn read_indices(tensor: &HostTensor) -> Cow<'_, [isize]> {
    match tensor.dtype() {
        #[cfg(target_pointer_width = "64")]
        DType::I64 => {
            const { assert!(size_of::<i64>() == size_of::<isize>()) };
            let data = tensor.storage::<i64>();
            Cow::Borrowed(bytemuck::cast_slice(data))
        }
        #[cfg(target_pointer_width = "32")]
        DType::I64 => Cow::Owned(
            tensor
                .storage::<i64>()
                .iter()
                .map(|&v| {
                    isize::try_from(v).unwrap_or_else(|_| {
                        panic!("read_indices: i64 index {v} out of isize range")
                    })
                })
                .collect(),
        ),
        #[cfg(target_pointer_width = "64")]
        DType::I32 => Cow::Owned(
            tensor
                .storage::<i32>()
                .iter()
                .map(|&v| v as isize)
                .collect(),
        ),
        #[cfg(target_pointer_width = "32")]
        DType::I32 => {
            const { assert!(size_of::<i32>() == size_of::<isize>()) };
            let data = tensor.storage::<i32>();
            Cow::Borrowed(bytemuck::cast_slice(data))
        }
        DType::I16 => Cow::Owned(
            tensor
                .storage::<i16>()
                .iter()
                .map(|&v| v as isize)
                .collect(),
        ),
        DType::I8 => Cow::Owned(tensor.storage::<i8>().iter().map(|&v| v as isize).collect()),
        DType::U64 => Cow::Owned(
            tensor
                .storage::<u64>()
                .iter()
                .map(|&v| {
                    isize::try_from(v).unwrap_or_else(|_| {
                        panic!("read_indices: u64 index {v} out of isize range")
                    })
                })
                .collect(),
        ),
        #[cfg(target_pointer_width = "64")]
        DType::U32 => Cow::Owned(
            tensor
                .storage::<u32>()
                .iter()
                .map(|&v| v as isize)
                .collect(),
        ),
        #[cfg(target_pointer_width = "32")]
        DType::U32 => Cow::Owned(
            tensor
                .storage::<u32>()
                .iter()
                .map(|&v| {
                    isize::try_from(v).unwrap_or_else(|_| {
                        panic!("read_indices: u32 index {v} out of isize range")
                    })
                })
                .collect(),
        ),
        DType::U16 => Cow::Owned(
            tensor
                .storage::<u16>()
                .iter()
                .map(|&v| v as isize)
                .collect(),
        ),
        DType::U8 => Cow::Owned(tensor.storage::<u8>().iter().map(|&v| v as isize).collect()),
        other => panic!("read_indices: unsupported index dtype {:?}", other),
    }
}

#[cold]
#[inline(never)]
fn index_oob(raw: isize, dim_size: usize) -> ! {
    panic!("index {raw} out of bounds for dimension of size {dim_size}");
}

/// Validate an index is non-negative and within bounds, panicking with a clear message otherwise.
#[inline(always)]
fn checked_index(raw: isize, dim_size: usize) -> usize {
    if raw < 0 || raw as usize >= dim_size {
        index_oob(raw, dim_size);
    }
    raw as usize
}

mod gather_values;
pub use gather_values::gather;
use gather_values::compute_gather_index;

mod scatter_values;
pub use scatter_values::*;

mod select_values;
pub use select_values::*;

mod select_add;
pub use select_add::*;

/// Compute row-major strides for a shape.
#[inline]
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let ndims = dims.len();
    let mut strides = vec![1usize; ndims];
    for i in (0..ndims.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

mod nd;
pub use nd::*;

// Type-specific wrappers

pub fn gather_f32(tensor: HostTensor, dim: usize, indices: HostTensor) -> HostTensor {
    gather::<f32>(tensor, dim, indices)
}

pub fn gather_f64(tensor: HostTensor, dim: usize, indices: HostTensor) -> HostTensor {
    gather::<f64>(tensor, dim, indices)
}

pub fn gather_i64(tensor: HostTensor, dim: usize, indices: HostTensor) -> HostTensor {
    gather::<i64>(tensor, dim, indices)
}

pub fn scatter_add_f32(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
    scatter_add::<f32>(tensor, dim, indices, value)
}

pub fn scatter_add_f64(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
    scatter_add::<f64>(tensor, dim, indices, value)
}

pub fn scatter_add_i64(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
    scatter_add::<i64>(tensor, dim, indices, value)
}

pub fn select_f32(tensor: HostTensor, dim: usize, indices: HostTensor) -> HostTensor {
    select::<f32>(tensor, dim, indices)
}

pub fn select_f64(tensor: HostTensor, dim: usize, indices: HostTensor) -> HostTensor {
    select::<f64>(tensor, dim, indices)
}

pub fn select_i64(tensor: HostTensor, dim: usize, indices: HostTensor) -> HostTensor {
    select::<i64>(tensor, dim, indices)
}

pub fn select_add_f32(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
    select_add::<f32>(tensor, dim, indices, value)
}

pub fn select_add_f64(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
    select_add::<f64>(tensor, dim, indices, value)
}

pub fn select_add_i64(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
    select_add::<i64>(tensor, dim, indices, value)
}

// Bool-specific operations

pub fn gather_bool(tensor: HostTensor, dim: usize, indices: HostTensor) -> HostTensor {
    gather::<u8>(tensor, dim, indices)
}

/// Scatter OR for bool tensors: ORs values into tensor at indexed positions.
pub fn scatter_or(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
    // Preserve the input tensor's bool dtype for the output.
    let out_dtype = ruda_core::tensor::BoolDType::from(tensor.dtype());
    let tensor = tensor.to_contiguous();
    let indices = indices.to_contiguous();
    let value = value.to_contiguous();

    let tensor_shape = tensor.layout().shape().clone();
    let indices_shape = indices.layout().shape();
    let value_shape = value.layout().shape();
    let ndims = tensor_shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );
    assert_eq!(
        indices_shape,
        value_shape,
        "scatter_or: indices shape {:?} must match value shape {:?}",
        indices_shape.to_vec(),
        value_shape.to_vec()
    );

    for i in 0..ndims {
        if i != dim {
            assert_eq!(
                tensor_shape[i], indices_shape[i],
                "scatter_or: shape mismatch at dim {}: tensor {} vs indices {}",
                i, tensor_shape[i], indices_shape[i]
            );
        }
    }

    let tensor_data: &[u8] = tensor.storage();
    let indices_data = read_indices(&indices);
    let value_data: &[u8] = value.storage();

    let mut result: Vec<u8> = tensor_data.to_vec();

    let tensor_strides = compute_strides(&tensor_shape);
    let indices_strides = compute_strides(indices_shape);

    let num_elements = indices_shape.num_elements();

    let scatter_or_dim_size = tensor_shape[dim];

    // Use 2D specialized path
    if ndims == 2 {
        let tensor_cols = tensor_shape[1];
        let indices_rows = indices_shape[0];
        let indices_cols = indices_shape[1];

        if dim == 0 {
            for i in 0..indices_rows {
                for j in 0..indices_cols {
                    let idx = i * indices_cols + j;
                    let dst_row = checked_index(indices_data[idx], scatter_or_dim_size);
                    result[dst_row * tensor_cols + j] |= value_data[idx];
                }
            }
        } else {
            for i in 0..indices_rows {
                for j in 0..indices_cols {
                    let idx = i * indices_cols + j;
                    let dst_col = checked_index(indices_data[idx], scatter_or_dim_size);
                    result[i * tensor_cols + dst_col] |= value_data[idx];
                }
            }
        }
    } else {
        let dim_stride = tensor_strides[dim];
        for idx in 0..num_elements {
            let index_val = checked_index(indices_data[idx], scatter_or_dim_size);
            let dst_idx = compute_gather_index(
                idx,
                index_val,
                dim,
                dim_stride,
                &indices_strides,
                &tensor_strides,
                ndims,
            );
            result[dst_idx] |= value_data[idx];
        }
    }

    crate::comparison::make_bool_tensor(result, tensor_shape, out_dtype)
}

// Tests kept here probe flex-specific behavior: non-I64 index dtype
// acceptance through the internal `read_indices` path and the uninit-
// buffer + rayon-chunked `select` kernel. Plain gather/scatter/select
// tests (including an empty-indices edge case) live in
// crates/ruda-backend-tests/tests/tensor/float/ops/{gather_scatter,select}.rs
// so every backend is exercised. When adding new tests, keep them here
// only if they probe flex internals; otherwise add them there.
#[cfg(test)]
mod tests;

pub mod dispatch_float;
pub mod dispatch_int;
