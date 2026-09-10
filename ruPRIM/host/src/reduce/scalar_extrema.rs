use super::*;

// ============================================================================
// Max / Min (all elements)
// ============================================================================

/// Max of all elements, returning a scalar tensor of shape \[1\].
pub fn max(tensor: HostTensor) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => max_f32_reduce(&tensor),
        DType::F64 => max_impl::<f64>(&tensor),
        DType::F16 => reduce_scalar_half(
            &tensor,
            f32::max,
            f32::NEG_INFINITY,
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => reduce_scalar_half(
            &tensor,
            f32::max,
            f32::NEG_INFINITY,
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I8 => max_impl::<i8>(&tensor),
        DType::I16 => max_impl::<i16>(&tensor),
        DType::I32 => max_impl::<i32>(&tensor),
        DType::I64 => max_impl::<i64>(&tensor),
        DType::U8 => max_impl::<u8>(&tensor),
        DType::U16 => max_impl::<u16>(&tensor),
        DType::U32 => max_impl::<u32>(&tensor),
        DType::U64 => max_impl::<u64>(&tensor),
        _ => panic!("max: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Min of all elements, returning a scalar tensor of shape \[1\].
pub fn min(tensor: HostTensor) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => min_f32_reduce(&tensor),
        DType::F64 => min_impl::<f64>(&tensor),
        DType::F16 => {
            reduce_scalar_half(&tensor, f32::min, f32::INFINITY, f16::to_f32, f16::from_f32)
        }
        DType::BF16 => reduce_scalar_half(
            &tensor,
            f32::min,
            f32::INFINITY,
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I8 => min_impl::<i8>(&tensor),
        DType::I16 => min_impl::<i16>(&tensor),
        DType::I32 => min_impl::<i32>(&tensor),
        DType::I64 => min_impl::<i64>(&tensor),
        DType::U8 => min_impl::<u8>(&tensor),
        DType::U16 => min_impl::<u16>(&tensor),
        DType::U32 => min_impl::<u32>(&tensor),
        DType::U64 => min_impl::<u64>(&tensor),
        _ => panic!("min: unsupported dtype {:?}", tensor.dtype()),
    }
}

fn max_f32_reduce(tensor: &HostTensor) -> HostTensor {
    let result = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[f32] = tensor.storage();
            max_f32_contiguous(&data[start..end])
        }
        None => {
            let data: &[f32] = tensor.storage();
            let elem_count = tensor.layout().num_elements();
            if data.len() == elem_count {
                // Non-contiguous but uses all elements (e.g., transposed)
                max_f32_contiguous(data)
            } else {
                StridedIter::new(tensor.layout())
                    .map(|idx| data[idx])
                    .reduce(|a, b| if a >= b { a } else { b })
                    .expect("max: tensor must not be empty")
            }
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    HostTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::F32)
}

#[inline]
fn max_f32_contiguous(data: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        kernels::max_f32(data)
    }

    #[cfg(not(feature = "simd"))]
    {
        data.iter()
            .copied()
            .reduce(|a, b| if a >= b { a } else { b })
            .expect("max: tensor must not be empty")
    }
}

fn min_f32_reduce(tensor: &HostTensor) -> HostTensor {
    let result = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[f32] = tensor.storage();
            min_f32_contiguous(&data[start..end])
        }
        None => {
            let data: &[f32] = tensor.storage();
            let elem_count = tensor.layout().num_elements();
            if data.len() == elem_count {
                min_f32_contiguous(data)
            } else {
                StridedIter::new(tensor.layout())
                    .map(|idx| data[idx])
                    .reduce(|a, b| if a <= b { a } else { b })
                    .expect("min: tensor must not be empty")
            }
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    HostTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::F32)
}

#[inline]
fn min_f32_contiguous(data: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        kernels::min_f32(data)
    }

    #[cfg(not(feature = "simd"))]
    {
        data.iter()
            .copied()
            .reduce(|a, b| if a <= b { a } else { b })
            .expect("min: tensor must not be empty")
    }
}

fn max_impl<E: Element + bytemuck::Pod + PartialOrd>(tensor: &HostTensor) -> HostTensor {
    let result: E = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[E] = tensor.storage();
            data[start..end]
                .iter()
                .copied()
                .reduce(|a, b| if a >= b { a } else { b })
                .expect("max: tensor must not be empty")
        }
        None => {
            let data: &[E] = tensor.storage();
            StridedIter::new(tensor.layout())
                .map(|idx| data[idx])
                .reduce(|a, b| if a >= b { a } else { b })
                .expect("max: tensor must not be empty")
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    HostTensor::new(
        bytes,
        Layout::contiguous(Shape::from(vec![1])),
        tensor.dtype(),
    )
}

fn min_impl<E: Element + bytemuck::Pod + PartialOrd>(tensor: &HostTensor) -> HostTensor {
    let result: E = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[E] = tensor.storage();
            data[start..end]
                .iter()
                .copied()
                .reduce(|a, b| if a <= b { a } else { b })
                .expect("min: tensor must not be empty")
        }
        None => {
            let data: &[E] = tensor.storage();
            StridedIter::new(tensor.layout())
                .map(|idx| data[idx])
                .reduce(|a, b| if a <= b { a } else { b })
                .expect("min: tensor must not be empty")
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    HostTensor::new(
        bytes,
        Layout::contiguous(Shape::from(vec![1])),
        tensor.dtype(),
    )
}

