#![allow(missing_docs)] // pub ruda modules

use crate::dsl::prelude::*;
use crate::dsl::{
    calculate_ruda_count_elemwise,
    ir::{ElemType, FloatKind, IntKind},
};
use ruda_core::{e2m1x2, e4m3, e5m2};
use crate::dsl::{tensor_vector_size_parallel};
use ruda_core::ir::{features::TypeUsage};

use crate::quantization::{
    layout::{ScalesView, scales_view_with_shape as scales_view},
    scheme::{QuantLevel, QuantMode, QuantScheme, QuantStore, QuantValue},
};
use crate::library::tensor::{
    View,
    layout::linear::{LinearView, linear_view},
};
use crate::library::quant::unpack_quantized;

/// Dequantize a vector of values into floating-point values using the provided scale.
#[ruda]
pub fn dequantize_symmetric<F: Float, FS: RudaPrimitive, N: Size>(
    value: Vector<F, N>,
    scale: FS,
) -> Vector<F, N> {
    // x = scale * x_q
    Vector::cast_from(scale) * value
}

/// Dequantize the value at a specified position using the provided quantization scheme.
///
/// Returns a vector of floating-point values. The number of values in the vector depends on the number of packed
/// values in the stored quantization type.
#[ruda]
pub fn dequantize_symmetric_packed_values<
    F: Float,
    NF: Size,
    FS: RudaPrimitive,
    QI: Int,
    NQ: Size,
>(
    position: usize,
    values: &View<Vector<QI, NQ>, usize>,
    scales: &View<FS, usize>,
    #[comptime] scheme: QuantScheme,
) -> Array<Vector<F, NF>> {
    dequantize_symmetric_packed_value_at::<F, NF, FS, QI, NQ>(
        position,
        values[position],
        scales,
        scheme,
    )
}

/// Dequantize a single value using the scale at the specified position.
///
/// Returns a vector of floating-point values. The number of values in the vector depends on the number of packed
/// values in the stored quantization type.
#[ruda]
pub fn dequantize_symmetric_packed_value_at<
    F: Float,
    NF: Size,
    FS: RudaPrimitive,
    QI: Int,
    NQ: Size,
>(
    position: usize,
    values: Vector<QI, NQ>,
    scales: &View<FS, usize>,
    #[comptime] scheme: QuantScheme,
) -> Array<Vector<F, NF>> {
    dequantize_symmetric_packed_value::<F, NF, FS, QI, NQ>(values, scales, position, scheme)
}

/// Dequantize a single packed value using the scale provided.
///
/// Returns a vector of floating-point values. The number of values in the vector depends on the number of packed
/// values in the stored quantization type.
#[ruda]
pub fn dequantize_symmetric_packed_value<
    F: Float,
    NF: Size,
    FS: RudaPrimitive,
    QS: Int,
    NQ: Size,
>(
    values: Vector<QS, NQ>,
    scales: &View<FS, usize>,
    position: usize,
    #[comptime] scheme: QuantScheme,
) -> Array<Vector<F, NF>> {
    let vector_size_values = values.vector_size();
    let num_quants = scheme.num_quants();
    let mut tmp = Array::new(vector_size_values);

    #[unroll]
    for i in 0..vector_size_values {
        let floats = unpack_q::<F, NF, QS>(values[i], scheme.value, scheme.store);
        let scale = scales[(position * vector_size_values) + i * num_quants];
        let values = dequantize_symmetric::<F, FS, NF>(floats, scale);
        tmp[i] = values;
    }

    tmp
}

/// Unpack a quantized integer into a vector of floating-point values, according to the specified quantization input type.
///
/// This handles types where multiple quantized values are packed into a single integer (the stored quantization type).
#[allow(clippy::explicit_counter_loop)]
#[ruda]
fn unpack_q<F: Float, NF: Size, QS: Int>(
    value: QS,
    #[comptime] quant: QuantValue,
    #[comptime] store: QuantStore,
) -> Vector<F, NF> {
    let size_quant = quant.size_bits();
    let size_store = store.size_bits(&quant);
    let num_quant = size_store / size_quant;

    let mut output = Vector::empty();

    let mask = QS::from_int((1 << size_quant) - 1);
    let sign_bit = QS::from_int(1 << (size_quant - 1));
    let two_pow_n = 1 << size_quant;

    #[unroll]
    for position in 0..num_quant {
        let offset = QS::cast_from(position * size_quant);
        let raw = (value >> offset) & mask;

        // Branchless two's complement conversion
        // If raw >= 2^(n-1), then result = raw - 2^n
        let raw_i32 = i32::cast_from(raw);
        let is_negative = i32::cast_from(raw >= sign_bit); // 1 if negative, 0 if positive
        let signed_value = raw_i32 - (is_negative * two_pow_n);

        output[position] = F::cast_from(signed_value);
    }

    output
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn dequantize_symmetric_packed_kernel<F: Float, NF: Size, FS: Numeric, NQ: Size>(
    input: &LinearView<Vector<u32, NQ>>,
    scales: &ScalesView<FS>,
    output: &mut LinearView<Vector<F, NF>, ReadWrite>,
    #[comptime] scheme: QuantScheme,
    #[define(F, FS)] _dtypes: [StorageType; 2],
) {
    if !input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let vector_size_in = input.vector_size();
    let vector_size_out = output.vector_size();

    comptime! {
        assert_eq!(vector_size_out, scheme.num_quants());
    }

    let values = input[ABSOLUTE_POS];
    #[unroll]
    for i in 0..vector_size_in {
        let decoded = unpack_quantized::<u32, F, Const<1>, NF>(
            Vector::cast_from(values[i]), scheme,
        );
        let output_pos = ABSOLUTE_POS * vector_size_in + i;
        let mut result = Vector::<F, NF>::empty();
        #[unroll]
        for lane in 0..vector_size_out {
            result[lane] = F::cast_from(scales[output_pos * vector_size_out + lane])
                * decoded[lane];
        }
        output[output_pos] = result;
    }
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn dequantize_symmetric_packed_strided_kernel<F: Float, FS: Numeric>(
    input: &LinearView<u32>,
    scales: &ScalesView<FS>,
    output: &mut LinearView<F, ReadWrite>,
    inner: usize,
    axis_len: usize,
    #[comptime] scheme: QuantScheme,
    #[define(F, FS)] _dtypes: [StorageType; 2],
) {
    if !input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let num_quants = scheme.num_quants();
    let packed_axis_len = axis_len / num_quants
        + usize::cast_from(axis_len % num_quants != 0);
    let group = ABSOLUTE_POS / inner;
    let inner_pos = ABSOLUTE_POS % inner;
    let outer_pos = group / packed_axis_len;
    let packed_axis_pos = group % packed_axis_len;
    let output_pos =
        (outer_pos * axis_len + packed_axis_pos * num_quants) * inner + inner_pos;
    let size!(NF) = num_quants;
    let decoded = unpack_quantized::<u32, F, Const<1>, NF>(
        Vector::cast_from(input[ABSOLUTE_POS]), scheme,
    );
    #[unroll]
    for lane in 0..num_quants {
        if packed_axis_pos * num_quants + lane < axis_len {
            let pos = output_pos + lane * inner;
            output[pos] = F::cast_from(scales[pos]) * decoded[lane];
        }
    }
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn dequantize_symmetric_native_kernel<F: Float, NF: Size, FS: Numeric, Q: Numeric, NQ: Size>(
    input: &LinearView<Vector<Q, NQ>>,
    scale: &ScalesView<FS>,
    output: &mut LinearView<Vector<F, NF>, ReadWrite>,
    #[define(F, FS, Q)] _dtypes: [StorageType; 3],
) {
    if !input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let native_packing = Q::packing_factor();
    let in_pos = ABSOLUTE_POS * input.vector_size() * native_packing;
    let values = Vector::<F, NF>::cast_from(input[ABSOLUTE_POS]);
    let mut result = Vector::<F, NF>::empty();
    #[unroll]
    for lane in 0..NF::value() {
        result[lane] = F::cast_from(scale[in_pos + lane]) * values[lane];
    }
    output[ABSOLUTE_POS] = result;
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn dequantize_symmetric_fp4_native_kernel<
    F: Float,
    NF: Size,
    FS: Numeric,
    NQ: Size,
>(
    input: &LinearView<Vector<e2m1x2, NQ>>,
    scale: &ScalesView<FS>,
    output: &mut LinearView<Vector<F, NF>, ReadWrite>,
    #[define(F, FS)] _dtypes: [StorageType; 2],
) {
    if !input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let values = Vector::<F, NF>::cast_from(input[ABSOLUTE_POS]);
    let mut result = Vector::<F, NF>::empty();
    #[unroll]
    for lane in 0..NF::value() {
        result[lane] = F::cast_from(scale[ABSOLUTE_POS * 2 + lane]) * values[lane];
    }
    output[ABSOLUTE_POS] = result;
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn dequantize_symmetric_fp4_native_strided_kernel<
    F: Float,
    NI: Size,
    FS: Numeric,
    NQ: Size,
>(
    input: &LinearView<Vector<e2m1x2, NI>>,
    scale: &ScalesView<FS>,
    output: &mut LinearView<Vector<F, NQ>, ReadWrite>,
    inner: usize,
    axis_len: usize,
    #[define(F, FS)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let packed_axis_len = (axis_len + 1) / 2;
    let group = ABSOLUTE_POS / inner;
    let inner_pos = ABSOLUTE_POS % inner;
    let outer_pos = group / axis_len;
    let axis_pos = group % axis_len;
    let input_pos =
        (outer_pos * packed_axis_len + axis_pos / 2) * inner + inner_pos;
    let size!(NPAIR) = 2;
    let unpacked = Vector::<F, NPAIR>::cast_from(input[input_pos]);
    let mut result = Vector::<F, NQ>::empty();
    if axis_pos.is_multiple_of(2) {
        result[0] = unpacked[0] * F::cast_from(scale[ABSOLUTE_POS]);
    } else {
        result[0] = unpacked[1] * F::cast_from(scale[ABSOLUTE_POS]);
    }
    output[ABSOLUTE_POS] = result;
}

#[allow(clippy::result_large_err)]
/// Convert the tensor back to a higher precision data type.
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    values: TensorBinding<R>,
    output: TensorBinding<R>,
    params: TensorBinding<R>,
    scheme: &QuantScheme,
    input_dtype: StorageType,
) -> Result<(), LaunchError> {
    let dtype_scale: StorageType = ElemType::from_quant_param(scheme.param).into();

    match scheme {
        QuantScheme {
            store: QuantStore::PackedU32(_),
            ..
        } => dequantize_packed(
            client,
            values,
            *scheme,
            params,
            output,
            input_dtype,
            dtype_scale,
        ),
        QuantScheme {
            value: QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2,
            store: QuantStore::Native,
            ..
        }
        | QuantScheme {
            value: QuantValue::E2M1,
            store: QuantStore::PackedNative(_),
            ..
        } => {
            let supported_uses = match scheme.value {
                QuantValue::E4M3 => e4m3::supported_uses(client),
                QuantValue::E5M2 => e5m2::supported_uses(client),
                QuantValue::E2M1 => e2m1x2::supported_uses(client),
                _ => i8::supported_uses(client),
            };
            if !supported_uses.contains(TypeUsage::Conversion) {
                panic!(
                    "{:?} is not supported for native quantization",
                    scheme.value
                );
            }

            dequantize_native(
                client,
                values,
                *scheme,
                params,
                output,
                input_dtype,
                dtype_scale,
            )
        }
        QuantScheme {
            store: QuantStore::Native | QuantStore::PackedNative(_),
            value,
            ..
        } => {
            panic!("{value:?} is not supported for native quantization");
        }
    }
}

fn dequantize_packed<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    scheme: QuantScheme,
    scale: TensorBinding<R>,
    output: TensorBinding<R>,
    input_dtype: StorageType,
    scale_dtype: StorageType,
) -> Result<(), LaunchError> {
    let logical_shape = output.shape.clone();
    let num_elems_input: usize = input.shape.iter().product();
    let packed_axis = match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            store: QuantStore::PackedU32(dim),
            mode: QuantMode::Symmetric,
            ..
        } => logical_shape.len() - 1 - dim,
        _ => panic!("Unsupported quantization scheme {scheme:?}"),
    };
    if logical_shape.contains(&0) {
        return Ok(());
    }
    let num_quants = scheme.num_quants();
    let axis_len = logical_shape[packed_axis];
    let inner = logical_shape[packed_axis + 1..].iter().product::<usize>();
    let can_vectorize = packed_axis == logical_shape.len() - 1
        && tensor_vector_size_parallel(
            core::iter::once(num_quants),
            &output.shape,
            &output.strides,
            packed_axis,
        ) == num_quants;
    let vector_size_in = if can_vectorize {
        tensor_vector_size_parallel(
            client.io_optimized_vector_sizes(size_of::<u32>()),
            &input.shape,
            &input.strides,
            packed_axis,
        )
    } else {
        1
    };

    let num_elems = num_elems_input / vector_size_in as usize;
    let ruda_dim = RudaDim::new(client.properties(), num_elems);
    let ruda_count = calculate_ruda_count_elemwise(client, num_elems, ruda_dim);
    let address_type = input
        .required_address_type(size_of::<u32>())
        .max(scale.required_address_type(scale_dtype.size()))
        .max(output.required_address_type(input_dtype.size()));

    if can_vectorize {
        unsafe {
            dequantize_symmetric_packed_kernel::launch_unchecked(
                client,
                ruda_count,
                ruda_dim,
                address_type,
                num_quants,
                vector_size_in,
                linear_view(input.clone()),
                scales_view(&logical_shape, scale, 1, &scheme),
                linear_view(output),
                scheme,
                [input_dtype, scale_dtype],
            )
        }
    } else {
        unsafe {
            dequantize_symmetric_packed_strided_kernel::launch_unchecked(
                client,
                ruda_count,
                ruda_dim,
                address_type,
                linear_view(input),
                scales_view(&logical_shape, scale, 1, &scheme),
                linear_view(output),
                inner,
                axis_len,
                scheme,
                [input_dtype, scale_dtype],
            )
        }
    }

    Ok(())
}

fn dequantize_native<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    scheme: QuantScheme,
    scale: TensorBinding<R>,
    output: TensorBinding<R>,
    input_dtype: StorageType,
    scale_dtype: StorageType,
) -> Result<(), LaunchError> {
    let logical_shape = output.shape.clone();
    let num_elems: usize = input.shape.iter().product();
    if logical_shape.contains(&0) {
        return Ok(());
    }

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            value: QuantValue::E2M1,
            store: QuantStore::PackedNative(packed_dim),
            ..
        } => {
            let output_elems: usize = output.shape.iter().product();
            let packed_axis = output.shape.len() - 1 - packed_dim;
            let axis_len = output.shape[packed_axis];
            let inner = output.shape[packed_axis + 1..].iter().product::<usize>();
            let address_type = input
                .required_address_type(size_of::<e2m1x2>())
                .max(scale.required_address_type(scale_dtype.size()))
                .max(output.required_address_type(input_dtype.size()));

            let can_vectorize = packed_axis == output.shape.len() - 1
                && tensor_vector_size_parallel(
                    core::iter::once(2),
                    &output.shape,
                    &output.strides,
                    packed_axis,
                ) == 2;
            if can_vectorize {
                let working_units = output_elems / 2;
                let ruda_dim = RudaDim::new(client.properties(), working_units);
                let ruda_count = calculate_ruda_count_elemwise(client, working_units, ruda_dim);
                unsafe {
                    dequantize_symmetric_fp4_native_kernel::launch_unchecked(
                        client,
                        ruda_count,
                        ruda_dim,
                        address_type,
                        2,
                        1,
                        linear_view(input.clone()),
                        scales_view(&logical_shape, scale.clone(), 1, &scheme),
                        linear_view(output.clone()),
                        [input_dtype, scale_dtype],
                    )
                };
            } else {
                let ruda_dim = RudaDim::new(client.properties(), output_elems);
                let ruda_count = calculate_ruda_count_elemwise(client, output_elems, ruda_dim);
                unsafe {
                    dequantize_symmetric_fp4_native_strided_kernel::launch_unchecked(
                        client,
                        ruda_count,
                        ruda_dim,
                        address_type,
                        1,
                        1,
                        linear_view(input.clone()),
                        scales_view(&logical_shape, scale, 1, &scheme),
                        linear_view(output),
                        inner,
                        axis_len,
                        [input_dtype, scale_dtype],
                    )
                };
            }
        }
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            value,
            store: QuantStore::Native,
            ..
        } => {
            let quant_dtype: ElemType = match value {
                QuantValue::Q8F | QuantValue::Q8S => ElemType::Int(IntKind::I8),
                QuantValue::E4M3 => ElemType::Float(FloatKind::E4M3),
                QuantValue::E5M2 => ElemType::Float(FloatKind::E5M2),
                QuantValue::E2M1 => ElemType::Float(FloatKind::E2M1),
                other => panic!("Unsupported quantization value {other:?}"),
            };
            let vector_size = tensor_vector_size_parallel(
                client.io_optimized_vector_sizes(input_dtype.size()).filter(|&size| {
                    tensor_vector_size_parallel(
                        core::iter::once(size),
                        &output.shape,
                        &output.strides,
                        output.shape.len() - 1,
                    ) == size
                }),
                &input.shape,
                &input.strides,
                input.shape.len() - 1,
            );
            let working_units = num_elems / vector_size;
            let ruda_dim = RudaDim::new(client.properties(), working_units);
            let ruda_count = calculate_ruda_count_elemwise(client, working_units, ruda_dim);

            let address_type = input
                .required_address_type(quant_dtype.size())
                .max(scale.required_address_type(scale_dtype.size()))
                .max(output.required_address_type(input_dtype.size()));

            unsafe {
                dequantize_symmetric_native_kernel::launch_unchecked(
                    client,
                    ruda_count,
                    ruda_dim,
                    address_type,
                    vector_size,
                    vector_size,
                    linear_view(input.clone()),
                    scales_view(&logical_shape, scale, 1, &scheme),
                    linear_view(output),
                    [input_dtype, scale_dtype, quant_dtype.into()],
                )
            }
        }
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    };

    Ok(())
}
