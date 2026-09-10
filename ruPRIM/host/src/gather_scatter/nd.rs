use super::*;

/// Multi-dimensional scatter: update `data` at locations specified by N-dimensional index tuples.
pub fn scatter_nd<
    E: Element + Pod + Default + Copy + core::ops::AddAssign + core::ops::Mul<Output = E> + PartialOrd,
>(
    data: HostTensor,
    indices: HostTensor,
    values: HostTensor,
    reduction: ruda_core::tensor::indexing::IndexingUpdateOp,
) -> HostTensor {
    use ruda_core::tensor::indexing::IndexingUpdateOp;

    let data = data.to_contiguous();
    let indices = indices.to_contiguous();
    let values = values.to_contiguous();

    let data_shape: Vec<usize> = data.layout().shape().to_vec();
    let idx_shape: Vec<usize> = indices.layout().shape().to_vec();
    let m = idx_shape.len();
    let k = idx_shape[m - 1];

    let num_indices: usize = idx_shape[..m - 1].iter().product();
    let slice_size: usize = data_shape[k..].iter().product();

    let data_data: &[E] = data.storage();
    let idx_data = read_indices(&indices);
    let val_data: &[E] = values.storage();

    let mut result: Vec<E> = data_data.to_vec();

    let strides = compute_strides(&data_shape);

    for n in 0..num_indices {
        let mut base_offset = 0usize;
        for j in 0..k {
            let idx_val = idx_data[n * k + j] as usize;
            base_offset += idx_val * strides[j];
        }

        let val_offset = n * slice_size;
        match reduction {
            IndexingUpdateOp::Assign => {
                result[base_offset..(base_offset + slice_size)]
                    .copy_from_slice(&val_data[val_offset..(val_offset + slice_size)]);
            }
            IndexingUpdateOp::Add => {
                for s in 0..slice_size {
                    result[base_offset + s] += val_data[val_offset + s];
                }
            }
            IndexingUpdateOp::Mul => {
                for s in 0..slice_size {
                    result[base_offset + s] = result[base_offset + s] * val_data[val_offset + s];
                }
            }
            IndexingUpdateOp::Min => {
                for s in 0..slice_size {
                    let b = val_data[val_offset + s];
                    if b < result[base_offset + s] {
                        result[base_offset + s] = b;
                    }
                }
            }
            IndexingUpdateOp::Max => {
                for s in 0..slice_size {
                    let b = val_data[val_offset + s];
                    if b > result[base_offset + s] {
                        result[base_offset + s] = b;
                    }
                }
            }
        }
    }

    let shape = Shape::from(data_shape);
    let bytes = Bytes::from_elems(result);
    HostTensor::new(bytes, Layout::contiguous(shape), E::dtype())
}

/// Multi-dimensional gather: collect slices from `data` at locations specified by N-dimensional
/// index tuples.
pub fn gather_nd<E: Element + Pod + Default + Copy>(
    data: HostTensor,
    indices: HostTensor,
) -> HostTensor {
    let data = data.to_contiguous();
    let indices = indices.to_contiguous();

    let data_shape: Vec<usize> = data.layout().shape().to_vec();
    let idx_shape: Vec<usize> = indices.layout().shape().to_vec();
    let m = idx_shape.len();
    let k = idx_shape[m - 1];

    let num_indices: usize = idx_shape[..m - 1].iter().product();
    let slice_size: usize = data_shape[k..].iter().product();

    let data_data: &[E] = data.storage();
    let idx_data = read_indices(&indices);

    let mut out_shape_vec: Vec<usize> = idx_shape[..m - 1].to_vec();
    out_shape_vec.extend_from_slice(&data_shape[k..]);

    let strides = compute_strides(&data_shape);

    let total = num_indices * slice_size;
    let mut result = vec![E::default(); total];

    for n in 0..num_indices {
        let mut base_offset = 0usize;
        for j in 0..k {
            let idx_val = idx_data[n * k + j] as usize;
            base_offset += idx_val * strides[j];
        }
        let out_offset = n * slice_size;
        result[out_offset..(out_offset + slice_size)]
            .copy_from_slice(&data_data[base_offset..(base_offset + slice_size)]);
    }

    let shape = Shape::from(out_shape_vec);
    let bytes = Bytes::from_elems(result);
    HostTensor::new(bytes, Layout::contiguous(shape), E::dtype())
}

