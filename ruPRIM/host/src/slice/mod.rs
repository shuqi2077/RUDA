//! Slice operations for FlexTensor.

use alloc::vec;
use alloc::vec::Vec;
use ruda_core::tensor::{DType, element::Element};
use ruda_core::{bytes::Bytes, tensor::{Shape, Slice}};
use half::{bf16, f16};

use ruda_core::tensor::host::{HostTensor, Layout};

/// Slice a tensor according to the given slice parameters.
///
/// For positive steps, this is zero-copy (metadata only).
/// For negative steps, data is copied to handle the reversal.
pub fn slice(tensor: HostTensor, slices: &[Slice]) -> HostTensor {
    let (new_layout, needs_copy) = tensor.layout().slice(slices);

    if !needs_copy {
        // Zero-copy: share data with new layout
        HostTensor::from_arc(tensor.data_arc(), new_layout, tensor.dtype())
    } else {
        // Needs copy due to negative steps
        slice_with_copy(&tensor, slices)
    }
}

/// Slice with data copy (handles negative steps).
fn slice_with_copy(tensor: &HostTensor, slices: &[Slice]) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => slice_copy_impl::<f32>(tensor, slices),
        DType::F64 => slice_copy_impl::<f64>(tensor, slices),
        DType::F16 => slice_copy_impl::<f16>(tensor, slices),
        DType::BF16 => slice_copy_impl::<bf16>(tensor, slices),
        DType::I32 => slice_copy_impl::<i32>(tensor, slices),
        DType::I64 => slice_copy_impl::<i64>(tensor, slices),
        DType::I16 => slice_copy_impl::<i16>(tensor, slices),
        DType::I8 => slice_copy_impl::<i8>(tensor, slices),
        DType::U32 => slice_copy_impl::<u32>(tensor, slices),
        DType::U64 => slice_copy_impl::<u64>(tensor, slices),
        DType::U16 => slice_copy_impl::<u16>(tensor, slices),
        DType::U8 => slice_copy_impl::<u8>(tensor, slices),
        DType::Bool(_) => slice_copy_impl::<u8>(tensor, slices),
        _ => panic!("slice: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Generic slice implementation with copy.
fn slice_copy_impl<E: Element + bytemuck::Pod + Default>(
    tensor: &HostTensor,
    slices: &[Slice],
) -> HostTensor {
    let src = tensor.storage::<E>();
    let src_layout = tensor.layout();
    let ndims = src_layout.num_dims();

    // Calculate output shape and collect normalized slice info
    let mut out_shape = Vec::with_capacity(ndims);
    let mut slice_info: Vec<(usize, usize, isize)> = Vec::with_capacity(ndims); // (start, len, step)

    for dim in 0..ndims {
        let dim_size = src_layout.shape()[dim] as isize;

        let slice = if dim < slices.len() {
            &slices[dim]
        } else {
            // Default: full range
            &Slice::new(0, None, 1)
        };

        let (start, len, step) = compute_slice_info(slice, dim_size);
        out_shape.push(len);
        slice_info.push((start, len, step));
    }

    let out_layout = Layout::contiguous(Shape::from(out_shape.clone()));
    let num_elements = out_layout.num_elements();

    if num_elements == 0 {
        let bytes = Bytes::from_elems::<E>(Vec::new());
        return HostTensor::new(bytes, out_layout, tensor.dtype());
    }

    // Allocate output
    let mut out_data: Vec<E> = Vec::with_capacity(num_elements);

    // Use recursive iteration for arbitrary dimensions
    let mut indices = vec![0usize; ndims];
    copy_slice_recursive(src, src_layout, &slice_info, &mut out_data, &mut indices, 0);

    let bytes = Bytes::from_elems(out_data);
    HostTensor::new(bytes, out_layout, tensor.dtype())
}

/// Recursively copy sliced elements.
fn copy_slice_recursive<E: Copy>(
    src: &[E],
    src_layout: &Layout,
    slice_info: &[(usize, usize, isize)],
    out: &mut Vec<E>,
    indices: &mut [usize],
    dim: usize,
) {
    let ndims = src_layout.num_dims();

    if dim == ndims {
        // Base case: copy single element
        let src_idx = compute_src_index(src_layout, slice_info, indices);
        out.push(src[src_idx]);
        return;
    }

    let (_, len, _) = slice_info[dim];

    for i in 0..len {
        indices[dim] = i;
        copy_slice_recursive(src, src_layout, slice_info, out, indices, dim + 1);
    }
}

/// Compute source index from output indices and slice info.
fn compute_src_index(
    layout: &Layout,
    slice_info: &[(usize, usize, isize)],
    out_indices: &[usize],
) -> usize {
    let mut idx = layout.start_offset() as isize;
    for (dim, &out_i) in out_indices.iter().enumerate() {
        let (start, _, step) = slice_info[dim];
        let src_i = if step > 0 {
            start + out_i * step as usize
        } else {
            // Negative step: start from high index, go down
            let result = start as isize - (out_i as isize) * (-step);
            debug_assert!(result >= 0, "slice: negative source index at dim {dim}");
            result as usize
        };
        idx += src_i as isize * layout.strides()[dim];
    }
    debug_assert!(idx >= 0, "slice: negative final index");
    idx as usize
}

/// Normalize a potentially negative index to a positive one.
fn normalize_index(idx: isize, dim_size: isize) -> usize {
    if idx < 0 {
        (dim_size + idx).max(0) as usize
    } else {
        idx as usize
    }
}

/// Assign values to a slice of a tensor.
pub fn slice_assign(tensor: HostTensor, slices: &[Slice], value: HostTensor) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => slice_assign_impl::<f32>(tensor, slices, value),
        DType::F64 => slice_assign_impl::<f64>(tensor, slices, value),
        DType::F16 => slice_assign_impl::<f16>(tensor, slices, value),
        DType::BF16 => slice_assign_impl::<bf16>(tensor, slices, value),
        DType::I32 => slice_assign_impl::<i32>(tensor, slices, value),
        DType::I64 => slice_assign_impl::<i64>(tensor, slices, value),
        DType::I16 => slice_assign_impl::<i16>(tensor, slices, value),
        DType::I8 => slice_assign_impl::<i8>(tensor, slices, value),
        DType::U32 => slice_assign_impl::<u32>(tensor, slices, value),
        DType::U64 => slice_assign_impl::<u64>(tensor, slices, value),
        DType::U16 => slice_assign_impl::<u16>(tensor, slices, value),
        DType::U8 => slice_assign_impl::<u8>(tensor, slices, value),
        DType::Bool(_) => slice_assign_impl::<u8>(tensor, slices, value),
        _ => panic!("slice_assign: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Generic slice assign implementation.
fn slice_assign_impl<E: Element + bytemuck::Pod + Clone>(
    tensor: HostTensor,
    slices: &[Slice],
    value: HostTensor,
) -> HostTensor {
    // Broadcast-scalar fast path: if `value` is a fully-broadcast
    // scalar (all strides zero), read the scalar once instead of
    // materializing the expansion via `to_contiguous`. The
    // `num_elements > 0` gate also guards the storage read against
    // zero-sized sources where `iter().all(...)` would be vacuously
    // true.
    if value.layout().num_elements() > 0 && value.layout().strides().iter().all(|&s| s == 0) {
        let scalar = value.storage::<E>()[value.layout().start_offset()];
        return slice_write_impl::<E>(tensor, slices, WriteSource::Scalar(scalar));
    }

    let value = value.to_contiguous();
    let val_src: &[E] = value.storage::<E>();
    slice_write_impl::<E>(tensor, slices, WriteSource::Slice(val_src))
}

/// Source to splat into a sliced region of a destination tensor. The
/// two variants drive the same dispatch tree; `Scalar` hits the
/// broadcast-scalar fast path (no value buffer), `Slice` hits the
/// memcpy-style assign path (advances through `val_src`).
#[derive(Copy, Clone)]
enum WriteSource<'a, E: Copy> {
    Scalar(E),
    Slice(&'a [E]),
}

impl<'a, E: Copy> WriteSource<'a, E> {
    /// Write a contiguous span of `dst` starting at `dst_offset` for
    /// `len` elements. For `Slice`, `src_offset` is the current
    /// position in the value buffer.
    #[inline]
    fn write_span(self, dst: &mut [E], dst_offset: usize, len: usize, src_offset: usize) {
        match self {
            WriteSource::Scalar(s) => dst[dst_offset..dst_offset + len].fill(s),
            WriteSource::Slice(src) => dst[dst_offset..dst_offset + len]
                .copy_from_slice(&src[src_offset..src_offset + len]),
        }
    }

    /// Write a single element. `src_idx` is only read in the `Slice`
    /// variant.
    #[inline]
    fn write_element(self, dst: &mut [E], dst_idx: usize, src_idx: usize) {
        match self {
            WriteSource::Scalar(s) => dst[dst_idx] = s,
            WriteSource::Slice(src) => dst[dst_idx] = src[src_idx],
        }
    }
}

/// Unified slice writer used by both `slice_assign_impl` and the
/// scalar-broadcast fast path. Walks the destination's sliced region
/// (1D / 2D inner-contig / ND inner-contig / strided fallback) and
/// pulls values from the given [`WriteSource`].
fn slice_write_impl<E: Element + bytemuck::Pod>(
    tensor: HostTensor,
    slices: &[Slice],
    source: WriteSource<'_, E>,
) -> HostTensor {
    let mut tensor = tensor.to_contiguous();
    let dst_layout = tensor.layout().clone();
    let ndims = dst_layout.num_dims();

    let slice_info: Vec<(usize, usize, isize)> = (0..ndims)
        .map(|dim| {
            let dim_size = dst_layout.shape()[dim] as isize;
            let slice = if dim < slices.len() {
                &slices[dim]
            } else {
                &Slice::new(0, None, 1)
            };
            compute_slice_info(slice, dim_size)
        })
        .collect();

    let dst = tensor.storage_mut::<E>();

    let inner_contiguous = slice_info
        .last()
        .map(|(_, _, step)| *step == 1)
        .unwrap_or(false);

    if ndims == 0 {
        // Rank 0: single scalar destination. Only reachable from the
        // scalar fast path; `slice_assign` on a rank-0 tensor with a
        // rank-0 source also ends up here.
        if !dst.is_empty() {
            source.write_element(dst, 0, 0);
        }
    } else if ndims == 1 {
        let (start, len, step) = slice_info[0];
        if step == 1 {
            source.write_span(dst, start, len, 0);
        } else {
            for i in 0..len {
                let dst_i = if step > 0 {
                    start + i * step as usize
                } else {
                    (start as isize - (i as isize) * (-step)) as usize
                };
                source.write_element(dst, dst_i, i);
            }
        }
    } else if ndims == 2 && inner_contiguous {
        let (row_start, row_len, row_step) = slice_info[0];
        let (col_start, col_len, _) = slice_info[1];
        let dst_cols = dst_layout.shape()[1];

        let mut val_offset = 0;
        for r in 0..row_len {
            let row_idx = if row_step > 0 {
                row_start + r * row_step as usize
            } else {
                (row_start as isize - (r as isize) * (-row_step)) as usize
            };
            let dst_row_start = row_idx * dst_cols + col_start;
            source.write_span(dst, dst_row_start, col_len, val_offset);
            val_offset += col_len;
        }
    } else if inner_contiguous {
        let inner_len = slice_info[ndims - 1].1;
        let outer_dims = ndims - 1;
        let dst_strides = dst_layout.strides();

        let outer_count: usize = slice_info.iter().take(outer_dims).map(|i| i.1).product();

        let mut outer_indices = vec![0usize; outer_dims];
        let mut val_offset = 0;

        for _ in 0..outer_count {
            let mut dst_offset = dst_layout.start_offset() as isize;
            for (dim, &idx) in outer_indices.iter().enumerate() {
                let (start, _, step) = slice_info[dim];
                let src_i = if step > 0 {
                    start + idx * step as usize
                } else {
                    (start as isize - (idx as isize) * (-step)) as usize
                };
                dst_offset += src_i as isize * dst_strides[dim];
            }
            dst_offset += slice_info[ndims - 1].0 as isize * dst_strides[ndims - 1];
            let dst_offset = dst_offset as usize;

            source.write_span(dst, dst_offset, inner_len, val_offset);
            val_offset += inner_len;

            // Odometer increment over outer dims.
            for dim in (0..outer_dims).rev() {
                outer_indices[dim] += 1;
                if outer_indices[dim] < slice_info[dim].1 {
                    break;
                }
                outer_indices[dim] = 0;
            }
        }
    } else {
        let total_elements: usize = slice_info.iter().map(|(_, len, _)| len).product();
        let dst_strides = dst_layout.strides();
        let mut indices = vec![0usize; ndims];

        for i in 0..total_elements {
            let mut dst_offset = dst_layout.start_offset() as isize;
            for (dim, &idx) in indices.iter().enumerate() {
                let (start, _, step) = slice_info[dim];
                let src_i = if step > 0 {
                    start + idx * step as usize
                } else {
                    (start as isize - (idx as isize) * (-step)) as usize
                };
                dst_offset += src_i as isize * dst_strides[dim];
            }

            source.write_element(dst, dst_offset as usize, i);

            for dim in (0..ndims).rev() {
                indices[dim] += 1;
                if indices[dim] < slice_info[dim].1 {
                    break;
                }
                indices[dim] = 0;
            }
        }
    }

    tensor
}

/// Compute slice info (start, len, step) for a dimension.
/// For negative step: start is the LAST index in the range (end-1), iterating down.
fn compute_slice_info(slice: &Slice, dim_size: isize) -> (usize, usize, isize) {
    let step = slice.step;
    let abs_step = step.unsigned_abs();

    // Normalize start and end to [0, dim_size]
    let range_start = normalize_index(slice.start, dim_size);
    let range_end = match slice.end {
        Some(e) => normalize_index(e, dim_size).min(dim_size as usize),
        None => dim_size as usize,
    };

    let len = if range_end > range_start {
        (range_end - range_start).div_ceil(abs_step)
    } else {
        0
    };

    if step > 0 {
        // Forward: start at low index, go up
        (range_start, len, step)
    } else {
        // Reverse: start at end-1 (highest index in range), go down
        // For s![2..8;-2]: start from index 7, go to 5, then 3
        let reverse_start = if range_end > range_start {
            range_end - 1
        } else {
            range_start
        };
        (reverse_start, len, step)
    }
}

// Tests kept here exercise flex-specific behavior: the internal
// `slice` / `slice_assign` helpers, the broadcast-scalar fast paths for
// `slice_fill` (1D contiguous, 2D inner-contig, 3D inner-contig, ND
// strided fallback, stepped-row 2D inner-contig), and non-f32 dtype
// coverage. General slice correctness across backends is covered by
// crates/ruda-backend-tests/tests/tensor/float/ops/{slice,slice_assign}.rs.
#[cfg(test)]
mod tests;
