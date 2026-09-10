use super::*;

/// Select add: adds values back to tensor at positions specified by 1D indices.
pub fn select_add<E: Element + Pod + Default + Copy + core::ops::AddAssign + Send + Sync>(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let indices = indices.to_contiguous();
    let value = value.to_contiguous();

    let tensor_shape = tensor.layout().shape().clone();
    let value_shape = value.layout().shape();
    let ndims = tensor_shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );
    assert_eq!(
        indices.layout().num_dims(),
        1,
        "select_add: indices must be 1D"
    );

    let tensor_data: &[E] = tensor.storage();
    let indices_data = read_indices(&indices);
    let value_data: &[E] = value.storage();
    let num_indices = indices_data.len();

    // Validate value shape
    for d in 0..ndims {
        if d == dim {
            assert_eq!(
                value_shape[d], num_indices,
                "select_add: value dim {} should be {} (num indices), got {}",
                d, num_indices, value_shape[d]
            );
        } else {
            assert_eq!(
                value_shape[d], tensor_shape[d],
                "select_add: value dim {} should match tensor dim {}, got {}",
                d, tensor_shape[d], value_shape[d]
            );
        }
    }

    let mut result: Vec<E> = tensor_data.to_vec();

    // Use optimized 2D implementation
    if ndims == 2 {
        select_add_2d(
            &mut result,
            &indices_data,
            value_data,
            tensor_shape[0],
            tensor_shape[1],
            num_indices,
            dim,
        );
        let bytes = Bytes::from_elems(result);
        return HostTensor::new(bytes, Layout::contiguous(tensor_shape), E::dtype());
    }

    // General N-D case
    let tensor_strides: Vec<usize> = compute_strides(&tensor_shape);
    let value_strides: Vec<usize> = compute_strides(value_shape);
    let select_add_dim_size = tensor_shape[dim];

    for (val_idx, &val) in value_data.iter().enumerate() {
        let mut remaining = val_idx;
        let mut dst_idx = 0;
        for d in 0..ndims {
            let coord = remaining / value_strides[d];
            remaining %= value_strides[d];
            if d == dim {
                let index_val = checked_index(indices_data[coord], select_add_dim_size);
                dst_idx += index_val * tensor_strides[d];
            } else {
                dst_idx += coord * tensor_strides[d];
            }
        }
        result[dst_idx] += val;
    }

    let bytes = Bytes::from_elems(result);
    HostTensor::new(bytes, Layout::contiguous(tensor_shape), E::dtype())
}

/// Optimized 2D select_add.
#[inline]
fn select_add_2d<E: Copy + core::ops::AddAssign>(
    result: &mut [E],
    indices_data: &[isize],
    value_data: &[E],
    tensor_rows: usize,
    tensor_cols: usize,
    num_indices: usize,
    dim: usize,
) {
    let dim_size = if dim == 0 { tensor_rows } else { tensor_cols };
    if dim == 0 {
        for (i, &idx) in indices_data.iter().enumerate() {
            let dst_row = checked_index(idx, dim_size);
            let dst_start = dst_row * tensor_cols;
            let src_start = i * tensor_cols;
            for j in 0..tensor_cols {
                result[dst_start + j] += value_data[src_start + j];
            }
        }
    } else {
        for row in 0..tensor_rows {
            for (j, &idx) in indices_data.iter().enumerate() {
                let dst_col = checked_index(idx, dim_size);
                result[row * tensor_cols + dst_col] += value_data[row * num_indices + j];
            }
        }
    }
}

