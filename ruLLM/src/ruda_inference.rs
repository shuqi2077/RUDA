#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use ruda_kernel::dsl as kernel_dsl;
use ruda_tensor::api::{DType, Tensor};
use ruda_core::ir::features::{Plane, TypeUsage};
use ruda_tensor::{DeviceOps, Shape, TensorPrimitive};
use ruda_kernel::tensor::contiguous::into_contiguous;
use ruda_kernel::tensor::allocation::empty_device_contiguous_dtype;
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda::runtime::server::ComputeServer;


// Capability checks use the selected device, never a global vendor-name guess.
fn specialized_dtype<R>(device: &R::Device, dtype: DType) -> bool
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
{
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32) { return false; }
    let client = R::client(device);
    let properties = client.properties();
    let storage: StorageType = dtype.into();
    let usage = properties.type_usage(storage);
    let f32_storage: StorageType = DType::F32.into();
    usage.contains(TypeUsage::Arithmetic) && usage.contains(TypeUsage::Buffer)
        && properties.type_usage(f32_storage).contains(TypeUsage::Arithmetic)
}

fn reduction_width<R>(device: &R::Device, dtype: DType, rows: usize) -> Option<u32>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
{
    if !specialized_dtype::<R>(device, dtype) { return None; }
    let client = R::client(device);
    let properties = client.properties();
    let hardware = &properties.hardware;
    if rows == 0 || rows > hardware.max_ruda_count.0 as usize { return None; }
    crate::runtime::fixed_reduction_width(
        hardware.plane_size_min, hardware.plane_size_max,
        properties.features.plane.contains(Plane::Ops),
        properties.features.plane.contains(Plane::Packed),
        hardware.max_ruda_dim.0, hardware.max_units_per_ruda,
    )
}

pub(crate) fn can_decode_specialized<R>(device: &R::Device, dtype: DType, shape: [usize; 4]) -> bool
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
{
    let [batch, heads, sequence, dimension] = shape;
    let Some(rows) = batch.checked_mul(heads) else { return false; };
    crate::runtime::supports_two_lane_decode(reduction_width::<R>(device, dtype, rows), dimension, sequence)
}

pub(crate) fn can_use_elementwise<R>(device: &R::Device, dtype: DType) -> bool
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
{
    specialized_dtype::<R>(device, dtype)
}

fn checked_u32(value: usize, name: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{name} exceeds the specialized kernel u32 interface"))
}

// Dedicated kernels use 32-bit metadata/index parameters. Check the reachable
// span, not just the logical element count (views can contain large holes).
fn check_kernel_span(shape: &[usize], strides: &[usize], name: &str) {
    assert_eq!(shape.len(), strides.len(), "{name}: shape/stride rank mismatch");
    for &value in shape.iter().chain(strides) { checked_u32(value, name); }
    if shape.contains(&0) { return; }
    let last = shape.iter().zip(strides).try_fold(0usize, |n, (&d, &stride)| {
        (d - 1).checked_mul(stride).and_then(|span| n.checked_add(span))
    }).expect("specialized tensor storage span overflow");
    checked_u32(last.checked_add(1).expect("specialized storage size overflow"), name);
}

fn checked_elements(shape: &[usize]) -> usize {
    if shape.contains(&0) { return 0; }
    shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
        .expect("specialized tensor element count overflow")
}

fn portable_rms_norm<R, F, I, BT, const D: usize>(
    input: Tensor<DeviceBackend<R, F, I, BT>, D>,
    gamma: Tensor<DeviceBackend<R, F, I, BT>, 1>, epsilon: f64,
) -> Tensor<DeviceBackend<R, F, I, BT>, D>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement,
{
    let dtype = input.dtype();
    let accumulation = if dtype == DType::F64 { DType::F64 } else { DType::F32 };
    let input = input.cast(accumulation);
    // Scale before squaring, so finite large inputs do not overflow the sum.
    let scale = input.clone().abs().max_dim(D - 1).clamp_min(epsilon.sqrt());
    let scaled = input / scale.clone();
    let rms = (scaled.clone().square().mean_dim(D - 1)
        + scale.clone().recip().mul_scalar(epsilon) / scale).sqrt();
    (scaled / rms).cast(dtype) * gamma.unsqueeze()
}

#[ruda(launch)]
fn rms_norm_kernel<F: Float>(
    input: &Array<F>,
    gamma: &Array<F>,
    output: &mut Array<F>,
    width: u32,
    epsilon: f32,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let width = width as usize;
    let row_offset = row * width;
    let mut sum = 0.0f32;
    let mut column = UNIT_POS_X as usize;

    while column < width {
        let value = f32::cast_from(input[row_offset + column]);
        sum += value * value;
        column += RUDA_DIM_X as usize;
    }

    sum = plane_sum(sum);
    if sum > 3.4028235e38f32 {
        let mut maximum = 0.0f32;
        let mut column = UNIT_POS_X as usize;
        while column < width {
            let magnitude = f32::cast_from(input[row_offset + column]).abs();
            if magnitude > maximum { maximum = magnitude; }
            column += RUDA_DIM_X as usize;
        }
        maximum = plane_max(maximum);
        let mut scaled_sum = 0.0f32;
        let mut column = UNIT_POS_X as usize;
        while column < width {
            let scaled = f32::cast_from(input[row_offset + column]) / maximum;
            scaled_sum += scaled * scaled;
            column += RUDA_DIM_X as usize;
        }
        scaled_sum = plane_sum(scaled_sum);
        let denominator = (scaled_sum / width as f32 + (epsilon / maximum) / maximum).sqrt();
        let mut column = UNIT_POS_X as usize;
        while column < width {
            let normalized = F::cast_from((f32::cast_from(input[row_offset + column]) / maximum) / denominator);
            output[row_offset + column] = normalized * gamma[column];
            column += RUDA_DIM_X as usize;
        }
    } else {
        let inverse_rms = 1.0f32 / (sum / width as f32 + epsilon).sqrt();
        let mut column = UNIT_POS_X as usize;
        while column < width {
            let normalized = F::cast_from(f32::cast_from(input[row_offset + column]) * inverse_rms);
            output[row_offset + column] = normalized * gamma[column];
            column += RUDA_DIM_X as usize;
        }
    }
}

#[ruda(launch)]
fn residual_rms_norm_kernel<F: Float>(
    residual: &kernel_dsl::prelude::Tensor<F>,
    update: &kernel_dsl::prelude::Tensor<F>,
    gamma: &Array<F>,
    hidden: &mut Array<F>,
    normalized: &mut Array<F>,
    width: u32,
    epsilon: f32,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let width = width as usize;
    let row_offset = row * width;
    let sequence = residual.shape(1);
    let batch_index = row / sequence;
    let sequence_index = row % sequence;
    let residual_row_offset =
        batch_index * residual.stride(0) + sequence_index * residual.stride(1);
    let update_row_offset = batch_index * update.stride(0) + sequence_index * update.stride(1);
    let mut sum = 0.0f32;
    let mut column = UNIT_POS_X as usize;

    while column < width {
        let position = row_offset + column;
        let value = residual[residual_row_offset + column * residual.stride(2)]
            + update[update_row_offset + column * update.stride(2)];
        hidden[position] = value;
        let value = f32::cast_from(value);
        sum += value * value;
        column += RUDA_DIM_X as usize;
    }

    sum = plane_sum(sum);
    if sum > 3.4028235e38f32 {
        let mut maximum = 0.0f32;
        let mut column = UNIT_POS_X as usize;
        while column < width {
            let magnitude = f32::cast_from(hidden[row_offset + column]).abs();
            if magnitude > maximum { maximum = magnitude; }
            column += RUDA_DIM_X as usize;
        }
        maximum = plane_max(maximum);
        let mut scaled_sum = 0.0f32;
        let mut column = UNIT_POS_X as usize;
        while column < width {
            let scaled = f32::cast_from(hidden[row_offset + column]) / maximum;
            scaled_sum += scaled * scaled;
            column += RUDA_DIM_X as usize;
        }
        scaled_sum = plane_sum(scaled_sum);
        let denominator = (scaled_sum / width as f32 + (epsilon / maximum) / maximum).sqrt();
        let mut column = UNIT_POS_X as usize;
        while column < width {
            let unit = F::cast_from((f32::cast_from(hidden[row_offset + column]) / maximum) / denominator);
            normalized[row_offset + column] = unit * gamma[column];
            column += RUDA_DIM_X as usize;
        }
    } else {
        let inverse_rms = 1.0f32 / (sum / width as f32 + epsilon).sqrt();
        let mut column = UNIT_POS_X as usize;
        while column < width {
            let position = row_offset + column;
            let value = hidden[position];
            let unit = F::cast_from(f32::cast_from(value) * inverse_rms);
            normalized[position] = unit * gamma[column];
            column += RUDA_DIM_X as usize;
        }
    }
}

#[ruda(launch)]
fn swiglu_kernel<F: Float>(
    gate_up: &Array<F>,
    output: &mut Array<F>,
    width: u32,
    #[define(F)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let width = width as usize;
    let output_position = ABSOLUTE_POS;
    let row = output_position / width;
    let column = output_position % width;
    let gate = gate_up[row * width * 2 + column];
    let up = gate_up[row * width * 2 + width + column];
    let value_f32 = f32::cast_from(gate);
    let silu = F::cast_from(value_f32 / (1.0f32 + (0.0f32 - value_f32).exp()));
    output[output_position] = silu * up;
}

#[ruda(launch)]
fn qkv_half_split_rope_kernel<F: Float>(
    projected: &kernel_dsl::prelude::Tensor<F>,
    frequencies: &kernel_dsl::prelude::Tensor<F>,
    query: &mut Array<F>,
    key: &mut Array<F>,
    value: &mut Array<F>,
    sequence: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dimension: u32,
    start_position: u32,
    cache_start: u32,
    key_capacity: u32,
    #[define(F)] _dtype: StorageType,
) {
    let position = ABSOLUTE_POS;
    let query_elements = query.len();
    let sequence = sequence as usize;
    let query_heads = query_heads as usize;
    let kv_heads = kv_heads as usize;
    let head_dimension = head_dimension as usize;
    let cache_start = cache_start as usize;
    let key_capacity = key_capacity as usize;
    let query_width = query_heads * head_dimension;
    let kv_width = kv_heads * head_dimension;
    let batch_size = query_elements / query_heads / sequence / head_dimension;
    let key_elements = batch_size * kv_heads * sequence * head_dimension;

    if position < query_elements {
        let dimension = position % head_dimension;
        let sequence_index = (position / head_dimension) % sequence;
        let head = (position / head_dimension / sequence) % query_heads;
        let batch = position / head_dimension / sequence / query_heads;
        let source_row = batch * projected.stride(0) + sequence_index * projected.stride(1);
        let source = source_row + (head * head_dimension + dimension) * projected.stride(2);
        let half = head_dimension / 2;
        let pair_dimension = dimension % half;
        let paired_dimension = if dimension < half {
            dimension + half
        } else {
            dimension - half
        };
        let paired = source_row + (head * head_dimension + paired_dimension) * projected.stride(2);
        let frequency_row = (start_position as usize + sequence_index) * frequencies.stride(0);
        let frequency = frequency_row + 2 * pair_dimension * frequencies.stride(1);
        let rotated = if dimension < half {
            F::new(0.0_f32) - projected[paired]
        } else {
            projected[paired]
        };
        query[position] = projected[source] * frequencies[frequency]
            + rotated * frequencies[frequency + frequencies.stride(2)];
    } else if position < query_elements + key_elements {
        let output_position = position - query_elements;
        let dimension = output_position % head_dimension;
        let sequence_index = (output_position / head_dimension) % sequence;
        let head = (output_position / head_dimension / sequence) % kv_heads;
        let batch = output_position / head_dimension / sequence / kv_heads;
        let source_row = batch * projected.stride(0) + sequence_index * projected.stride(1);
        let source =
            source_row + (query_width + head * head_dimension + dimension) * projected.stride(2);
        let half = head_dimension / 2;
        let pair_dimension = dimension % half;
        let paired_dimension = if dimension < half {
            dimension + half
        } else {
            dimension - half
        };
        let paired = source_row
            + (query_width + head * head_dimension + paired_dimension) * projected.stride(2);
        let frequency_row = (start_position as usize + sequence_index) * frequencies.stride(0);
        let frequency = frequency_row + 2 * pair_dimension * frequencies.stride(1);
        let rotated = if dimension < half {
            F::new(0.0_f32) - projected[paired]
        } else {
            projected[paired]
        };
        let cache_position =
            ((batch * kv_heads + head) * key_capacity + cache_start + sequence_index)
                * head_dimension
                + dimension;
        key[cache_position] = projected[source] * frequencies[frequency]
            + rotated * frequencies[frequency + frequencies.stride(2)];
    } else if position < query_elements + 2 * key_elements {
        let output_position = position - query_elements - key_elements;
        let dimension = output_position % head_dimension;
        let sequence_index = (output_position / head_dimension) % sequence;
        let head = (output_position / head_dimension / sequence) % kv_heads;
        let batch = output_position / head_dimension / sequence / kv_heads;
        let source_row = batch * projected.stride(0) + sequence_index * projected.stride(1);
        let source = source_row
            + (query_width + kv_width + head * head_dimension + dimension) * projected.stride(2);
        let cache_position =
            ((batch * kv_heads + head) * key_capacity + cache_start + sequence_index)
                * head_dimension
                + dimension;
        value[cache_position] = projected[source];
    }
}

#[ruda(launch)]
fn gqa_decode_attention_kernel<F: Float>(
    query: &Array<F>,
    key: &Array<F>,
    value: &Array<F>,
    output: &mut Array<F>,
    query_heads: u32,
    kv_heads: u32,
    key_sequence: u32,
    key_capacity: u32,
    head_dimension: u32,
    scale: f32,
    #[define(F)] _dtype: StorageType,
) {
    let query_heads = query_heads as usize;
    let kv_heads = kv_heads as usize;
    let key_sequence = key_sequence as usize;
    let key_capacity = key_capacity as usize;
    let head_dimension = head_dimension as usize;
    let row = RUDA_POS_X as usize;
    let batch = row / query_heads;
    let query_head = row % query_heads;
    let kv_head = query_head / (query_heads / kv_heads);
    let query_offset = (batch * query_heads + query_head) * head_dimension;
    let kv_offset = (batch * kv_heads + kv_head) * key_capacity * head_dimension;

    let lane = UNIT_POS_X as usize;
    let query_0 = f32::cast_from(query[query_offset + lane]);
    let query_1 = f32::cast_from(query[query_offset + lane + RUDA_DIM_X as usize]);
    let mut softmax_maximum = SharedMemory::<f32>::new(1usize);
    if UNIT_POS_X == 0 {
        softmax_maximum[0] = -3.4028235e38f32;
    }
    sync_ruda();

    // Match the generic path's numerical boundary: matmul produces F, scale
    // is applied in F, then softmax reads the score as f32.
    for sequence_index in 0..key_sequence {
        let key_offset = kv_offset + sequence_index * head_dimension;
        let mut dot = 0.0f32;
        dot += query_0 * f32::cast_from(key[key_offset + lane]);
        dot += query_1 * f32::cast_from(key[key_offset + lane + RUDA_DIM_X as usize]);
        let rounded_dot = f32::cast_from(F::cast_from(plane_sum(dot)));
        let score = f32::cast_from(F::cast_from(rounded_dot * scale));
        if UNIT_POS_X == 0 && score > softmax_maximum[0] {
            softmax_maximum[0] = score;
        }
        sync_ruda();
    }

    let maximum = softmax_maximum[0];
    let mut denominator = 0.0f32;
    for sequence_index in 0..key_sequence {
        let key_offset = kv_offset + sequence_index * head_dimension;
        let mut dot = 0.0f32;
        dot += query_0 * f32::cast_from(key[key_offset + lane]);
        dot += query_1 * f32::cast_from(key[key_offset + lane + RUDA_DIM_X as usize]);
        let rounded_dot = f32::cast_from(F::cast_from(plane_sum(dot)));
        let score = f32::cast_from(F::cast_from(rounded_dot * scale));
        denominator += (score - maximum).exp();
    }

    // The generic path casts normalized f32 probabilities back to F before
    // multiplying by V. Preserve that rounding boundary as well.
    let mut accumulator_0 = 0.0f32;
    let mut accumulator_1 = 0.0f32;
    for sequence_index in 0..key_sequence {
        let key_offset = kv_offset + sequence_index * head_dimension;
        let mut dot = 0.0f32;
        dot += query_0 * f32::cast_from(key[key_offset + lane]);
        dot += query_1 * f32::cast_from(key[key_offset + lane + RUDA_DIM_X as usize]);
        let rounded_dot = f32::cast_from(F::cast_from(plane_sum(dot)));
        let score = f32::cast_from(F::cast_from(rounded_dot * scale));
        let probability = f32::cast_from(F::cast_from((score - maximum).exp() / denominator));
        accumulator_0 += probability * f32::cast_from(value[key_offset + lane]);
        accumulator_1 +=
            probability * f32::cast_from(value[key_offset + lane + RUDA_DIM_X as usize]);
    }

    output[query_offset + lane] = F::cast_from(accumulator_0);
    output[query_offset + lane + RUDA_DIM_X as usize] = F::cast_from(accumulator_1);
}

pub(crate) fn rms_norm<R, F, I, BT, const D: usize>(
    input: Tensor<DeviceBackend<R, F, I, BT>, D>,
    gamma: Tensor<DeviceBackend<R, F, I, BT>, 1>,
    epsilon: f64,
) -> Tensor<DeviceBackend<R, F, I, BT>, D>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    assert!(D > 0, "RMSNorm requires a positive tensor rank");
    let shape = input.dims();
    let width = shape[D - 1];
    assert!(width > 0 && epsilon.is_finite() && epsilon > 0.0,
        "RMSNorm requires positive width and finite positive epsilon");
    assert!(input.dtype() == DType::F64 || ((epsilon as f32).is_finite() && epsilon as f32 > 0.0),
        "RMSNorm epsilon is not representable in the accumulation dtype");
    assert_eq!(gamma.dims(), [width]);
    assert_eq!(input.dtype(), gamma.dtype());
    assert_eq!(input.device(), gamma.device(), "RMSNorm tensors must share a device");
    let rows = checked_elements(&shape) / width;
    if rows == 0 { return input; }
    let subgroup = reduction_width::<R>(&input.device(), input.dtype(), rows);
    let Some(subgroup) = subgroup.filter(|_| width <= u32::MAX as usize && checked_elements(&shape) <= u32::MAX as usize) else {
        return portable_rms_norm(input, gamma, epsilon);
    };
    let input = into_contiguous(input.into_primitive().tensor());
    let gamma = into_contiguous(gamma.into_primitive().tensor());
    let width = input.meta.shape()[D - 1];
    assert_eq!(
        gamma.meta.num_elements(),
        width,
        "RMSNorm gamma width must match the input's final dimension"
    );
    assert_eq!(
        input.dtype, gamma.dtype,
        "RMSNorm input and gamma must have the same dtype"
    );
    let rows = input.meta.num_elements() / width;
    let output = empty_device_contiguous_dtype::<R>(
        input.client.clone(),
        input.device.clone(),
        input.meta.shape().clone(),
        input.dtype,
    );

    let client = input.client.clone();
    rms_norm_kernel::launch::<R>(
        &client,
        RudaCount::Static(checked_u32(rows, "RMSNorm rows"), 1, 1),
        RudaDim::new_1d(subgroup),
        input.into_array_arg(),
        gamma.into_array_arg(),
        output.clone().into_array_arg(),
        checked_u32(width, "RMSNorm width"),
        epsilon as f32,
        output.dtype.into(),
    );

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

pub(crate) fn residual_rms_norm<R, F, I, BT, const D: usize>(
    residual: Tensor<DeviceBackend<R, F, I, BT>, D>,
    update: Tensor<DeviceBackend<R, F, I, BT>, D>,
    gamma: Tensor<DeviceBackend<R, F, I, BT>, 1>,
    epsilon: f64,
) -> (
    Tensor<DeviceBackend<R, F, I, BT>, D>,
    Tensor<DeviceBackend<R, F, I, BT>, D>,
)
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    assert_eq!(
        D, 3,
        "internal residual RMSNorm expects rank-three Llama activations"
    );
    let shape = residual.dims();
    let width = shape[D - 1];
    assert!(width > 0 && epsilon.is_finite() && epsilon > 0.0,
        "residual RMSNorm requires positive width and finite positive epsilon");
    assert_eq!(update.dims(), shape);
    assert_eq!(gamma.dims(), [width]);
    assert_eq!(update.dtype(), residual.dtype());
    assert_eq!(gamma.dtype(), residual.dtype());
    assert_eq!(update.device(), residual.device());
    assert_eq!(gamma.device(), residual.device());
    assert!(residual.dtype() == DType::F64 || ((epsilon as f32).is_finite() && epsilon as f32 > 0.0),
        "residual RMSNorm epsilon is not representable in the accumulation dtype");
    let rows = checked_elements(&shape) / width;
    if rows == 0 { return (residual.clone(), residual); }
    let subgroup = reduction_width::<R>(&residual.device(), residual.dtype(), rows);
    let Some(subgroup) = subgroup.filter(|_| width <= u32::MAX as usize && checked_elements(&shape) <= u32::MAX as usize) else {
        let hidden = residual + update;
        let normalized = portable_rms_norm(hidden.clone(), gamma, epsilon);
        return (hidden, normalized);
    };
    let residual = residual.into_primitive().tensor();
    let update = update.into_primitive().tensor();
    let gamma = into_contiguous(gamma.into_primitive().tensor());
    let width = residual.meta.shape()[D - 1];
    assert_eq!(update.meta.shape(), residual.meta.shape());
    assert_eq!(gamma.meta.num_elements(), width);
    assert_eq!(update.dtype, residual.dtype);
    assert_eq!(gamma.dtype, residual.dtype);
    check_kernel_span(residual.meta.shape(), residual.meta.strides(), "residual storage");
    check_kernel_span(update.meta.shape(), update.meta.strides(), "update storage");
    let rows = residual.meta.num_elements() / width;
    let make_output = || {
        empty_device_contiguous_dtype::<R>(
            residual.client.clone(),
            residual.device.clone(),
            residual.meta.shape().clone(),
            residual.dtype,
        )
    };
    let hidden = make_output();
    let normalized = make_output();
    let client = residual.client.clone();

    residual_rms_norm_kernel::launch::<R>(
        &client,
        RudaCount::Static(checked_u32(rows, "residual RMSNorm rows"), 1, 1),
        RudaDim::new_1d(subgroup),
        residual.into_tensor_arg(),
        update.into_tensor_arg(),
        gamma.into_array_arg(),
        hidden.clone().into_array_arg(),
        normalized.clone().into_array_arg(),
        checked_u32(width, "residual RMSNorm width"),
        epsilon as f32,
        hidden.dtype.into(),
    );

    (
        Tensor::from_primitive(TensorPrimitive::Float(hidden)),
        Tensor::from_primitive(TensorPrimitive::Float(normalized)),
    )
}

pub(crate) fn swiglu<R, F, I, BT, const D: usize>(
    gate_up: Tensor<DeviceBackend<R, F, I, BT>, D>,
    width: usize,
) -> Tensor<DeviceBackend<R, F, I, BT>, D>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    assert!(D > 0 && width > 0, "SwiGLU requires a positive rank and width");
    let twice_width = width.checked_mul(2).expect("SwiGLU packed width overflow");
    let shape = gate_up.dims();
    assert_eq!(shape[D - 1], twice_width);
    let elements = checked_elements(&shape);
    if !specialized_dtype::<R>(&gate_up.device(), gate_up.dtype()) || width > u32::MAX as usize || elements > u32::MAX as usize || elements == 0 {
        let mut gate_range: [std::ops::Range<usize>; D] = std::array::from_fn(|i| 0..shape[i]);
        let mut up_range: [std::ops::Range<usize>; D] = std::array::from_fn(|i| 0..shape[i]);
        gate_range[D - 1] = 0..width;
        up_range[D - 1] = width..twice_width;
        let gate = gate_up.clone().slice(gate_range);
        let up = gate_up.slice(up_range);
        return ruda_tensor::api::activation::silu(gate) * up;
    }
    let gate_up = into_contiguous(gate_up.into_primitive().tensor());
    let mut output_shape = gate_up.meta.shape().clone();
    assert_eq!(
        output_shape[D - 1],
        width * 2,
        "packed SwiGLU input must contain equal gate and up halves"
    );
    output_shape[D - 1] = width;
    let output = empty_device_contiguous_dtype::<R>(
        gate_up.client.clone(),
        gate_up.device.clone(),
        output_shape,
        gate_up.dtype,
    );
    let ruda_dim = RudaDim::new(gate_up.client.properties(), output.meta.num_elements());
    let ruda_count =
        calculate_ruda_count_elemwise(&gate_up.client, output.meta.num_elements(), ruda_dim);

    let client = gate_up.client.clone();
    swiglu_kernel::launch::<R>(
        &client,
        ruda_count,
        ruda_dim,
        gate_up.into_array_arg(),
        output.clone().into_array_arg(),
        width as u32,
        output.dtype.into(),
    );

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

pub(crate) fn qkv_half_split_rope_cached<R, F, I, BT>(
    projected: Tensor<DeviceBackend<R, F, I, BT>, 3>,
    frequencies: Tensor<DeviceBackend<R, F, I, BT>, 3>,
    key: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    value: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    query_heads: usize,
    kv_heads: usize,
    head_dimension: usize,
    start_position: usize,
    cache_start: usize,
) -> (
    Tensor<DeviceBackend<R, F, I, BT>, 4>,
    Tensor<DeviceBackend<R, F, I, BT>, 4>,
    Tensor<DeviceBackend<R, F, I, BT>, 4>,
)
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    assert!(query_heads > 0 && kv_heads > 0 && query_heads % kv_heads == 0);
    assert!(head_dimension > 0 && head_dimension % 2 == 0, "RoPE head dimension must be positive and even");
    assert_eq!(frequencies.dtype(), projected.dtype());
    assert_eq!(frequencies.device(), projected.device());
    assert_eq!(key.device(), projected.device());
    assert_eq!(value.device(), projected.device());
    let projected = projected.into_primitive().tensor();
    let frequencies = frequencies.into_primitive().tensor();
    let key = into_contiguous(key.into_primitive().tensor());
    let value = into_contiguous(value.into_primitive().tensor());
    let [batch, sequence, projected_width] = projected.meta.shape().dims::<3>();
    let query_width = query_heads.checked_mul(head_dimension).expect("query width overflow");
    let kv_width = kv_heads.checked_mul(head_dimension).expect("KV width overflow");
    assert_eq!(projected_width, kv_width.checked_mul(2).and_then(|n| n.checked_add(query_width)).expect("packed QKV width overflow"));
    assert!(batch > 0 && sequence > 0);
    assert!(start_position.checked_add(sequence).is_some_and(|n| n <= frequencies.meta.shape()[0]));
    assert_eq!(frequencies.meta.shape()[1], head_dimension);
    assert_eq!(frequencies.meta.shape()[2], 2);
    let [key_batch, key_heads, key_capacity, key_dimension] = key.meta.shape().dims::<4>();
    assert_eq!(
        [key_batch, key_heads, key_dimension],
        [batch, kv_heads, head_dimension]
    );
    assert_eq!(value.meta.shape(), key.meta.shape());
    assert!(cache_start.checked_add(sequence).is_some_and(|n| n <= key_capacity));
    assert_eq!(key.dtype, projected.dtype);
    assert_eq!(value.dtype, projected.dtype);

    for (dimension, name) in [
        (sequence, "sequence"), (query_heads, "query_heads"), (kv_heads, "kv_heads"),
        (head_dimension, "head_dimension"), (start_position, "start_position"),
        (cache_start, "cache_start"), (key_capacity, "key_capacity"),
    ] { checked_u32(dimension, name); }

    check_kernel_span(projected.meta.shape(), projected.meta.strides(), "projected storage");
    check_kernel_span(frequencies.meta.shape(), frequencies.meta.strides(), "RoPE storage");
    check_kernel_span(key.meta.shape(), key.meta.strides(), "key storage");
    check_kernel_span(value.meta.shape(), value.meta.strides(), "value storage");
    let query_elements = checked_elements(&[batch, query_heads, sequence, head_dimension]);
    let kv_elements = checked_elements(&[batch, kv_heads, sequence, head_dimension]);
    let elements = kv_elements.checked_mul(2).and_then(|n| n.checked_add(query_elements)).expect("QKV launch size overflow");
    checked_u32(elements, "QKV launch elements");
    let make_output = |shape| {
        empty_device_contiguous_dtype::<R>(
            projected.client.clone(),
            projected.device.clone(),
            shape,
            projected.dtype,
        )
    };
    let query = make_output(Shape::new([batch, query_heads, sequence, head_dimension]));
    let ruda_dim = RudaDim::new(projected.client.properties(), elements);
    let ruda_count = calculate_ruda_count_elemwise(&projected.client, elements, ruda_dim);
    let client = projected.client.clone();

    qkv_half_split_rope_kernel::launch::<R>(
        &client,
        ruda_count,
        ruda_dim,
        projected.into_tensor_arg(),
        frequencies.into_tensor_arg(),
        query.clone().into_array_arg(),
        key.clone().into_array_arg(),
        value.clone().into_array_arg(),
        checked_u32(sequence, "sequence"),
        checked_u32(query_heads, "query_heads"),
        checked_u32(kv_heads, "kv_heads"),
        checked_u32(head_dimension, "head_dimension"),
        checked_u32(start_position, "start_position"),
        checked_u32(cache_start, "cache_start"),
        checked_u32(key_capacity, "key_capacity"),
        query.dtype.into(),
    );

    (
        Tensor::from_primitive(TensorPrimitive::Float(query)),
        Tensor::from_primitive(TensorPrimitive::Float(key)),
        Tensor::from_primitive(TensorPrimitive::Float(value)),
    )
}

pub(crate) fn gqa_decode_attention<R, F, I, BT>(
    query: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    key: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    value: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    key_sequence: usize,
) -> Tensor<DeviceBackend<R, F, I, BT>, 4>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    assert_eq!(query.dtype(), key.dtype());
    assert_eq!(query.dtype(), value.dtype());
    assert_eq!(query.device(), key.device());
    assert_eq!(query.device(), value.device());
    let [batch, heads, _, _] = query.dims();
    let rows = batch.checked_mul(heads).expect("decode row count overflow");
    let subgroup = reduction_width::<R>(&query.device(), query.dtype(), rows)
        .expect("specialized decode requires a fixed supported subgroup; use portable attention");
    let query = into_contiguous(query.into_primitive().tensor());
    let key = into_contiguous(key.into_primitive().tensor());
    let value = into_contiguous(value.into_primitive().tensor());
    let [batch, query_heads, query_sequence, head_dimension] = query.meta.shape().dims::<4>();
    let [key_batch, kv_heads, key_capacity, key_dimension] = key.meta.shape().dims::<4>();
    assert_eq!(query_sequence, 1, "decode attention requires one query");
    assert_eq!([key_batch, key_dimension], [batch, head_dimension]);
    assert_eq!(value.meta.shape(), key.meta.shape());
    assert!(key_sequence > 0 && key_sequence <= key_capacity);
    assert!(kv_heads > 0 && query_heads > 0 && query_heads.is_multiple_of(kv_heads));
    assert_eq!(
        head_dimension, subgroup as usize * 2,
        "decode kernel requires exactly two values per subgroup lane"
    );

    check_kernel_span(query.meta.shape(), query.meta.strides(), "query storage");
    check_kernel_span(key.meta.shape(), key.meta.strides(), "key storage");
    check_kernel_span(value.meta.shape(), value.meta.strides(), "value storage");
    for (dimension, name) in [(query_heads, "query_heads"), (kv_heads, "kv_heads"),
        (key_sequence, "key_sequence"), (key_capacity, "key_capacity"), (head_dimension, "head_dimension")] {
        checked_u32(dimension, name);
    }
    let output = empty_device_contiguous_dtype::<R>(
        query.client.clone(),
        query.device.clone(),
        query.meta.shape().clone(),
        query.dtype,
    );
    let client = query.client.clone();
    gqa_decode_attention_kernel::launch::<R>(
        &client,
        RudaCount::Static(checked_u32(rows, "decode rows"), 1, 1),
        RudaDim::new_1d(subgroup),
        query.into_array_arg(),
        key.into_array_arg(),
        value.into_array_arg(),
        output.clone().into_array_arg(),
        checked_u32(query_heads, "query_heads"),
        checked_u32(kv_heads, "kv_heads"),
        checked_u32(key_sequence, "key_sequence"),
        checked_u32(key_capacity, "key_capacity"),
        checked_u32(head_dimension, "head_dimension"),
        1.0f32 / (head_dimension as f32).sqrt(),
        output.dtype.into(),
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

#[cfg(all(test, feature = "nvidia"))]
mod numerical_boundary_tests {
    use super::*;
    use ruda_tensor::api::TensorData;
    use ruda_tensor_device::cuda::{Cuda, CudaDevice};
    use half::bf16;

    type B = Cuda<bf16, i32>;

    #[test]
    fn packed_bf16_operator_boundaries() {
        let device = CudaDevice::default();
        let width = 896;
        let round = |x: f32| bf16::from_f32(x).to_f32();
        let input: Vec<_> = (0..width).map(|i| bf16::from_f32((i as f32 - 448.0) / 113.0)).collect();
        let gamma: Vec<_> = (0..width).map(|i| bf16::from_f32(0.3 + (i % 31) as f32 / 17.0)).collect();
        let x = Tensor::<B, 3>::from_data(TensorData::new(input.clone(), [1, 1, width]), &device);
        let g = Tensor::<B, 1>::from_data(TensorData::new(gamma.clone(), [width]), &device);
        let norm = rms_norm(x.clone(), g.clone(), 1e-6).into_data().to_vec::<bf16>().unwrap();
        let (_, fused) = residual_rms_norm(x.clone(), Tensor::zeros_like(&x), g, 1e-6);
        assert_eq!(norm, fused.into_data().to_vec::<bf16>().unwrap());
        let sum: f32 = input.iter().map(|v| v.to_f32().powi(2)).sum();
        let inverse = 1.0 / (sum / width as f32 + 1e-6).sqrt();
        for i in 0..width {
            assert_eq!(norm[i].to_f32(), round(round(input[i].to_f32() * inverse) * gamma[i].to_f32()), "RMSNorm {i}");
        }
        let packed = Tensor::<B, 3>::from_data(TensorData::new([input.clone(), gamma.clone()].concat(), [1, 1, width * 2]), &device);
        let actual = swiglu(packed, width).into_data().to_vec::<bf16>().unwrap();
        for i in 0..width {
            let gate = input[i].to_f32();
            let expected = round(round(gate / (1.0 + (-gate).exp())) * gamma[i].to_f32());
            assert_eq!(actual[i].to_f32(), expected, "SwiGLU {i}");
        }
    }
}

#[cfg(all(test, any(feature = "nvidia", feature = "amd")))]
mod backend_tests;
