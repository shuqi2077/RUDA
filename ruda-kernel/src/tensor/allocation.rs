use super::element::TensorElement;
use ruda_core::bytes::Bytes;
use ruda_core::tensor::{DType, Shape, Metadata, QParamTensor, QuantScheme, QuantParam, QuantValue, params_shape};
use ruda_core::quant::scheme::QuantStore;
use crate::dsl::{Runtime, e2m1x2, client::ComputeClient, server::{MemoryLayout, MemoryLayoutDescriptor, MemoryLayoutStrategy}};
use super::{RudaTensor, QParams};

/// Create a tensor with uninitialized memory
pub fn empty_device<R: Runtime, E: TensorElement>(
    client: ComputeClient<R>,
    device: R::Device,
    shape: Shape,
) -> RudaTensor<R> {
    let MemoryLayout { memory, strides } = client.empty_tensor(shape.clone(), size_of::<E>());

    RudaTensor::new(
        client,
        memory,
        Metadata::new(shape, strides),
        device,
        E::dtype(),
    )
}

/// Create a tensor with uninitialized memory
pub fn empty_device_dtype<R: Runtime>(
    client: ComputeClient<R>,
    device: R::Device,
    shape: Shape,
    dtype: DType,
) -> RudaTensor<R> {
    let MemoryLayout { memory, strides } = client.empty_tensor(shape.clone(), dtype.size());

    RudaTensor::new(client, memory, Metadata::new(shape, strides), device, dtype)
}

/// Create a quantized tensor with packed values (u32).
pub fn new_qtensor_optimized<R: Runtime>(
    data: Bytes,
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
) -> RudaTensor<R> {
    new_qtensor(data, shape, scheme, device, MemoryLayoutStrategy::Optimized)
}

/// Create a quantized tensor with packed values (u32).
fn new_qtensor<R: Runtime>(
    data: Bytes,
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
    kind: MemoryLayoutStrategy,
) -> RudaTensor<R> {
    new_quantized(shape, scheme, device, Some(data), kind)
}

/// Create an empty quantized tensor.
pub fn empty_qtensor_optimized<R: Runtime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
) -> RudaTensor<R> {
    empty_qtensor(shape, scheme, device, MemoryLayoutStrategy::Optimized)
}

/// Create an empty quantized tensor.
pub fn empty_qtensor<R: Runtime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
    kind: MemoryLayoutStrategy,
) -> RudaTensor<R> {
    new_quantized(shape, scheme, device, None, kind)
}

fn new_quantized<R: Runtime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
    data: Option<Bytes>,
    alloc_kind: MemoryLayoutStrategy,
) -> RudaTensor<R> {
    let client = R::client(device);
    let shape: Shape = shape.into();
    let mut shape_value: Shape = shape.clone();

    let rank = shape.rank();
    let num_quants = scheme.num_quants();

    let data_size = match scheme.store {
        QuantStore::PackedU32(packed_dim) => {
            let packed_axis = rank - packed_dim - 1;
            shape_value[packed_axis] = shape_value[packed_axis].div_ceil(num_quants);
            size_of::<u32>()
        }
        QuantStore::Native => match scheme.value {
            QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2 => {
                size_of::<i8>()
            }
            QuantValue::Q4F
            | QuantValue::Q4S
            | QuantValue::Q2F
            | QuantValue::Q2S
            | QuantValue::E2M1 => {
                panic!("Can't store native sub-byte values")
            }
        },
        QuantStore::PackedNative(packed_dim) => match scheme.value {
            QuantValue::E2M1 => {
                let packed_axis = rank - packed_dim - 1;
                shape_value[packed_axis] = shape_value[packed_axis].div_ceil(num_quants);
                size_of::<e2m1x2>()
            }
            other => panic!("{other:?} doesn't support native packing"),
        },
    };

    let scales_dtype = match scheme.param {
        QuantParam::F32 => DType::F32,
        QuantParam::F16 => DType::F16,
        QuantParam::BF16 => DType::BF16,
        // Represented by U8 and reinterpreted in the kernel
        QuantParam::UE8M0 | QuantParam::UE4M3 => DType::U8,
    };

    let scales_shape = params_shape(&shape, scheme.level);
    let data_desc = MemoryLayoutDescriptor::new(alloc_kind, shape_value.clone(), data_size);
    let scales_desc =
        MemoryLayoutDescriptor::new(alloc_kind, scales_shape.clone(), scales_dtype.size());

    let mut tensors = match data {
        Some(data) => {
            let num_bytes = shape_value.num_elements() * data_size;

            match data.split(num_bytes) {
                Ok((bytes_data, bytes_scales)) => client
                    .create_tensors(vec![(data_desc, bytes_data), (scales_desc, bytes_scales)]),
                Err((data, _)) => client.create_tensors_from_slices(vec![
                    (data_desc, &data[..num_bytes]),
                    (scales_desc, &data[num_bytes..]),
                ]),
            }
        }
        None => client.empty_tensors(vec![data_desc, scales_desc]),
    };
    let MemoryLayout {
        memory: scales_handle,
        strides: scales_strides,
    } = tensors.remove(1);
    let MemoryLayout { memory, strides } = tensors.remove(0);

    let scales = QParamTensor {
        offset_start: scales_handle.offset_start.unwrap_or(0) as usize,
        offset_end: scales_handle.offset_end.unwrap_or(0) as usize,
        metadata: Metadata::new(scales_shape, scales_strides),
        dtype: scales_dtype,
    };
    let qparams = QParams { scales };

    RudaTensor::new_quantized(
        client,
        memory,
        shape,
        device.clone(),
        strides,
        DType::QFloat(scheme),
        qparams,
    )
}


/// Create a contiguous tensor with uninitialized memory
pub fn empty_device_contiguous_dtype<R: Runtime>(
    client: ComputeClient<R>,
    device: R::Device,
    shape: Shape,
    dtype: DType,
) -> RudaTensor<R> {
    let descriptor = MemoryLayoutDescriptor::contiguous(shape.clone(), dtype.size());
    let MemoryLayout { memory, strides } = client.empty_tensors(vec![descriptor]).remove(0);

    RudaTensor::new(client, memory, Metadata::new(shape, strides), device, dtype)
}

pub fn empty<R: Runtime>(
    shape: Shape,
    device: &R::Device,
    dtype: DType,
) -> RudaTensor<R> {
    let client = R::client(device);
    let alloc = client.empty_tensor(shape.clone(), dtype.size());

    RudaTensor::new(
        client,
        alloc.memory,
        Metadata::new(shape, alloc.strides),
        device.clone(),
        dtype,
    )
}
