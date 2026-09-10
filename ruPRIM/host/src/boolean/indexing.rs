use alloc::{vec, vec::Vec};
use ruda_core::{bytes::Bytes, tensor::{DType, IntDType, Shape, host::{HostTensor, Layout}}};

pub fn bool_select_or(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
    let mut result = crate::gather_scatter::select_add::<u8>(tensor, dim, indices, value);
    // Clamp to 0/1: select_add sums u8 values, but bool OR saturates at 1
    let storage: &mut [u8] = result.storage_mut();
    for v in storage.iter_mut() {
        if *v > 1 {
            *v = 1;
        }
    }
    result
}

pub async fn bool_argwhere(tensor: HostTensor, out_dtype: IntDType) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let ndims = shape.num_dims();
    let data: &[u8] = tensor.storage();
    let n = shape.num_elements();

    let count = data[..n].iter().filter(|&&v| v != 0).count();
    let mut coords: Vec<isize> = Vec::with_capacity(count * ndims);
    let strides = ruda_core::tensor::host::layout::contiguous_strides_usize(&shape);

    for (flat_idx, &val) in data[..n].iter().enumerate() {
        if val != 0 {
            let mut remaining = flat_idx;
            for &s in &strides {
                coords.push((remaining / s) as isize);
                remaining %= s;
            }
        }
    }

    let out_shape = Shape::from(vec![count, ndims]);
    let result = HostTensor::new(
        Bytes::from_elems(coords),
        Layout::contiguous(out_shape),
        ruda_core::tensor::host::dtype::INDEX_DTYPE,
    );
    if result.dtype() != DType::from(out_dtype) {
        crate::cast::int_cast(result, out_dtype)
    } else {
        result
    }
}

