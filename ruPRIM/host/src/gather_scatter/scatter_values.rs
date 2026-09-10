use super::*;

/// Scatter add: adds values to tensor at positions specified by indices.
pub fn scatter_add<E: Element + Pod + Default + Copy + core::ops::AddAssign + Send + Sync>(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
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
        "scatter_add: indices shape {:?} must match value shape {:?}",
        indices_shape.to_vec(),
        value_shape.to_vec()
    );

    for i in 0..ndims {
        if i != dim {
            assert_eq!(
                tensor_shape[i], indices_shape[i],
                "scatter_add: shape mismatch at dim {}: tensor {} vs indices {}",
                i, tensor_shape[i], indices_shape[i]
            );
        }
    }

    let tensor_data: &[E] = tensor.storage();
    let indices_data = read_indices(&indices);
    let value_data: &[E] = value.storage();

    let mut result: Vec<E> = tensor_data.to_vec();

    let tensor_strides: Vec<usize> = compute_strides(&tensor_shape);
    let indices_strides: Vec<usize> = compute_strides(indices_shape);

    let num_elements = indices_shape.num_elements();

    // Use specialized 2D implementation
    if ndims == 2 {
        scatter_add_2d(
            &mut result,
            &indices_data,
            value_data,
            tensor_shape[0],
            tensor_shape[1],
            indices_shape[0],
            indices_shape[1],
            dim,
        );
    } else {
        // General N-D case (sequential due to potential index conflicts)
        let dim_stride = tensor_strides[dim];
        let scatter_dim_size = tensor_shape[dim];
        for idx in 0..num_elements {
            let index_val = checked_index(indices_data[idx], scatter_dim_size);
            let dst_idx = compute_gather_index(
                idx,
                index_val,
                dim,
                dim_stride,
                &indices_strides,
                &tensor_strides,
                ndims,
            );
            result[dst_idx] += value_data[idx];
        }
    }

    let bytes = Bytes::from_elems(result);
    HostTensor::new(bytes, Layout::contiguous(tensor_shape), E::dtype())
}

/// Optimized 2D scatter_add implementation.
#[inline]
#[allow(clippy::too_many_arguments)]
fn scatter_add_2d<E: Copy + core::ops::AddAssign>(
    result: &mut [E],
    indices_data: &[isize],
    value_data: &[E],
    tensor_rows: usize,
    tensor_cols: usize,
    indices_rows: usize,
    indices_cols: usize,
    dim: usize,
) {
    let dim_size = if dim == 0 { tensor_rows } else { tensor_cols };
    if dim == 0 {
        for i in 0..indices_rows {
            for j in 0..indices_cols {
                let idx = i * indices_cols + j;
                let dst_row = checked_index(indices_data[idx], dim_size);
                result[dst_row * tensor_cols + j] += value_data[idx];
            }
        }
    } else {
        for i in 0..indices_rows {
            for j in 0..indices_cols {
                let idx = i * indices_cols + j;
                let dst_col = checked_index(indices_data[idx], dim_size);
                result[i * tensor_cols + dst_col] += value_data[idx];
            }
        }
    }
}

