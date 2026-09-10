use super::{HostTensor, Layout};
use crate::{bytes::Bytes, tensor::{DType, element::Element}};
use bytemuck::Pod;

/// Cast a tensor from half-precision type E to f32.
pub fn cast_to_f32<E: Element + Pod + Copy>(
    tensor: HostTensor,
    to_f32: fn(E) -> f32,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let data: &[E] = tensor.storage();
    let f32_data: alloc::vec::Vec<f32> = data.iter().map(|&v| to_f32(v)).collect();
    let bytes = Bytes::from_elems(f32_data);
    HostTensor::new(bytes, Layout::contiguous(shape), DType::F32)
}

/// Cast a tensor from f32 back to half-precision type E.
pub fn cast_from_f32<E: Element + Pod + Copy>(
    tensor: HostTensor,
    from_f32: fn(f32) -> E,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let data: &[f32] = tensor.storage();
    let half_data: alloc::vec::Vec<E> = data.iter().map(|&v| from_f32(v)).collect();
    let bytes = Bytes::from_elems(half_data);
    HostTensor::new(bytes, Layout::contiguous(shape), E::dtype())
}

