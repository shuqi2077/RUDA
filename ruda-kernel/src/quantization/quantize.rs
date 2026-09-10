use crate::dsl::{calculate_ruda_count_elemwise, ir::ElemType, prelude::*, tensor_vector_size_parallel};
use crate::library::{tensor::into_contiguous, tensor::layout::linear::LinearView, tensor::{View, layout::linear::linear_view}};
use ruda_core::ir::{features::TypeUsage};
use ruda_core::{e2m1x2, e4m3, e5m2};

use crate::quantization::{
    layout::{ScalesLayout, scales_view_with_shape as scales_view},
};
use crate::quantization::{
    layout::{ScalesView, scales_layout_with_shape as scales_layout},
    scheme::{QuantLevel, QuantMode, QuantScheme, QuantStore, QuantValue},
};

#[ruda]
fn quantize_symmetric<F: Float, N: Size, FS: RudaPrimitive>(
    value: Vector<F, N>,
    scale: FS,
    range_min: F,
    range_max: F,
    #[comptime] quant: QuantValue,
) -> Vector<F, N> {
    let scaled = value / Vector::cast_from(scale);
    let rounded = match quant {
        QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1 => scaled,
        _ => Vector::round(scaled),
    };
    clamp(
        rounded,
        Vector::new(range_min),
        Vector::new(range_max),
    )
}

#[ruda]
fn quantize_symmetric_q<F: Float, N: Size, FS: RudaPrimitive, Q: Scalar>(
    value: Vector<F, N>,
    scale: FS,
    range_min: F,
    range_max: F,
    #[comptime] quant: QuantValue,
) -> Vector<Q, N> {
    Vector::cast_from(quantize_symmetric::<F, N, FS>(
        value, scale, range_min, range_max, quant,
    ))
}

/// Pack a vector of quantized floating-point values into a single integer (the stored quantization type),
/// according to the specified quantization input type.
#[allow(clippy::explicit_counter_loop)]
#[ruda]
fn pack_q<F: Float, N: Size, QS: Int>(value: Vector<F, N>, #[comptime] quant: QuantValue) -> QS {
    match quant {
        QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1 => {
            let size!(NB) = QS::type_size_bits().comptime() / 8;
            let bytes = match quant {
                QuantValue::E4M3 => Vector::<u8, NB>::reinterpret(Vector::<e4m3, NB>::cast_from(value)),
                QuantValue::E5M2 => Vector::<u8, NB>::reinterpret(Vector::<e5m2, NB>::cast_from(value)),
                _ => Vector::<u8, NB>::reinterpret(Vector::<e2m1x2, NB>::cast_from(value)),
            };
            let mut packed = QS::from_int(0);
            #[unroll]
            for byte in 0..NB::value() {
                packed |= QS::cast_from(bytes[byte]) << QS::cast_from(byte * 8);
            }
            packed
        }
        _ => {
            let size_quant = quant.size_bits();

            let size_store = QS::type_size_bits().comptime();
            let num_quants = size_store / size_quant;

            let mask = (1 << size_quant) - 1;
            let mut packed = QS::from_int(0);

            // Shift and combine into QS (using i32 for sign extension)
            #[unroll]
            for position in 0..num_quants {
                let offset = QS::cast_from(position * size_quant);
                let shifted = QS::cast_from(i32::cast_from(value[position]) & mask) << offset;
                packed |= shifted;
            }

            packed
        }
    }
}

#[ruda]
fn write_scale<SI: Numeric, FS: RudaPrimitive>(
    in_pos: usize,
    scale: &View<SI, usize>,
    out_scale: &mut View<FS, usize, ReadWrite>,
    scales_layout: &ScalesLayout,
) -> FS {
    let scale = FS::cast_from(scale[in_pos]);

    // Write the scale into the output buffer
    if scales_layout.is_block_start(in_pos) {
        out_scale[in_pos] = scale;
    }

    scale
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn quantize_symmetric_native_kernel<F: Float, N: Size, SI: Numeric, FS: Numeric, Q: Numeric>(
    input: &LinearView<Vector<F, N>>,
    scale: &ScalesView<SI>,
    range_min: InputScalar,
    range_max: InputScalar,
    output: &mut LinearView<Vector<Q, N>, ReadWrite>,
    out_scale: &mut ScalesView<FS, ReadWrite>,
    scales_layout: ScalesLayout,
    #[comptime] quant: QuantValue,
    #[define(F, SI, FS, Q)] _dtypes: [StorageType; 4],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let native_packing = Q::packing_factor();
    let in_pos = ABSOLUTE_POS * input.vector_size() * native_packing;
    let values = input[ABSOLUTE_POS];
    let mut quantized = Vector::<Q, N>::empty();
    #[unroll]
    for lane in 0..N::value() {
        let scale = write_scale(in_pos + lane, scale, out_scale, &scales_layout);
        let value = quantize_symmetric_q::<F, Const<1>, FS, Q>(
            Vector::cast_from(values[lane]),
            scale,
            range_min.get::<F>(),
            range_max.get::<F>(),
            quant,
        );
        quantized[lane] = value[0];
    }
    output[ABSOLUTE_POS] = quantized;
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn quantize_symmetric_packed_kernel<F: Float, N: Size, SI: Numeric, FS: Numeric>(
    input: &LinearView<Vector<F, N>>,
    scale: &ScalesView<SI>,
    range_min: InputScalar,
    range_max: InputScalar,
    output: &mut LinearView<u32, ReadWrite>,
    out_scale: &mut ScalesView<FS, ReadWrite>,
    scales_layout: ScalesLayout,
    inner: usize,
    axis_len: usize,
    #[comptime] scheme: QuantScheme,
    #[define(F, SI, FS)] _dtypes: [StorageType; 3],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let num_quants = scheme.num_quants();
    let size!(NQ) = num_quants;
    let mut quantized = Vector::<F, NQ>::new(F::new(0.0_f32));

    if input.vector_size().comptime() == num_quants {
        let values = input[ABSOLUTE_POS];
        let packed_pos = ABSOLUTE_POS * num_quants;
        #[unroll]
        for lane in 0..num_quants {
            let scale = write_scale(packed_pos + lane, scale, out_scale, &scales_layout);
            let value = quantize_symmetric::<F, Const<1>, FS>(
                Vector::cast_from(values[lane]),
                scale,
                range_min.get::<F>(),
                range_max.get::<F>(),
                scheme.value,
            );
            quantized[lane] = value[0];
        }
    } else {
        let packed_axis_len = axis_len / num_quants
            + usize::cast_from(axis_len % num_quants != 0);
        let group = ABSOLUTE_POS / inner;
        let inner_pos = ABSOLUTE_POS % inner;
        let outer_pos = group / packed_axis_len;
        let packed_axis_pos = group % packed_axis_len;
        let input_pos =
            (outer_pos * axis_len + packed_axis_pos * num_quants) * inner + inner_pos;
        #[unroll]
        for lane in 0..num_quants {
            if packed_axis_pos * num_quants + lane < axis_len {
                let pos = input_pos + lane * inner;
                let scale = write_scale(pos, scale, out_scale, &scales_layout);
                let value = quantize_symmetric::<F, Const<1>, FS>(
                    Vector::cast_from(input[pos][0]),
                    scale,
                    range_min.get::<F>(),
                    range_max.get::<F>(),
                    scheme.value,
                );
                quantized[lane] = value[0];
            }
        }
    }
    output[ABSOLUTE_POS] = pack_q::<F, NQ, u32>(quantized, scheme.value);
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn quantize_symmetric_fp4_native_kernel<
    F: Float,
    NF: Size,
    SI: Numeric,
    FS: Numeric,
    NQ: Size,
>(
    input: &LinearView<Vector<F, NF>>,
    scale: &ScalesView<SI>,
    range_min: InputScalar,
    range_max: InputScalar,
    output: &mut LinearView<Vector<e2m1x2, NQ>, ReadWrite>,
    out_scale: &mut ScalesView<FS, ReadWrite>,
    scales_layout: ScalesLayout,
    #[define(F, SI, FS)] _dtypes: [StorageType; 3],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let in_pos = ABSOLUTE_POS * 2;
    let values = input[ABSOLUTE_POS];
    let mut quantized = Vector::<F, NF>::empty();
    #[unroll]
    for lane in 0..NF::value() {
        let scale = write_scale(in_pos + lane, scale, out_scale, &scales_layout);
        let value = quantize_symmetric::<F, Const<1>, FS>(
            Vector::cast_from(values[lane]),
            scale,
            range_min.get::<F>(),
            range_max.get::<F>(),
            QuantValue::E2M1,
        );
        quantized[lane] = value[0];
    }
    output[ABSOLUTE_POS] = Vector::<e2m1x2, NQ>::cast_from(quantized);
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn quantize_symmetric_fp4_native_strided_kernel<
    F: Float,
    NI: Size,
    SI: Numeric,
    FS: Numeric,
    NQ: Size,
>(
    input: &LinearView<Vector<F, NI>>,
    scale: &ScalesView<SI>,
    range_min: InputScalar,
    range_max: InputScalar,
    output: &mut LinearView<Vector<e2m1x2, NQ>, ReadWrite>,
    out_scale: &mut ScalesView<FS, ReadWrite>,
    scales_layout: ScalesLayout,
    inner: usize,
    axis_len: usize,
    #[define(F, SI, FS)] _dtypes: [StorageType; 3],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let packed_axis_len = (axis_len + 1) / 2;
    let group = ABSOLUTE_POS / inner;
    let inner_pos = ABSOLUTE_POS % inner;
    let outer_pos = group / packed_axis_len;
    let packed_axis_pos = group % packed_axis_len;
    let input_pos =
        (outer_pos * axis_len + packed_axis_pos * 2) * inner + inner_pos;
    let size!(NPAIR) = 2;
    let mut quantized = Vector::<F, NPAIR>::new(F::new(0.0_f32));
    #[unroll]
    for lane in 0..2 {
        if packed_axis_pos * 2 + lane < axis_len {
            let pos = input_pos + lane * inner;
            let scale = write_scale(pos, scale, out_scale, &scales_layout);
            let value = quantize_symmetric::<F, Const<1>, FS>(
                Vector::cast_from(input[pos][0]),
                scale,
                range_min.get::<F>(),
                range_max.get::<F>(),
                QuantValue::E2M1,
            );
            quantized[lane] = value[0];
        }
    }
    output[ABSOLUTE_POS] = Vector::<e2m1x2, NQ>::cast_from(quantized);
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn copy_scales_kernel<SI: Numeric, FS: Numeric>(
    input: &LinearView<SI>,
    output: &mut LinearView<FS, ReadWrite>,
    #[define(SI, FS)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    output[ABSOLUTE_POS] = FS::cast_from(input[ABSOLUTE_POS]);
}

fn copy_empty_scales<R: Runtime>(
    client: &ComputeClient<R>,
    scale: TensorBinding<R>,
    out_scale: TensorBinding<R>,
    input_dtype: ElemType,
    output_dtype: ElemType,
) -> Result<(), LaunchError> {
    let working_units = out_scale.shape.iter().product();
    if working_units == 0 {
        return Ok(());
    }
    let ruda_dim = RudaDim::new(client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(client, working_units, ruda_dim);
    let address_type = scale.required_address_type(input_dtype.size())
        .max(out_scale.required_address_type(output_dtype.size()));
    unsafe {
        copy_scales_kernel::launch_unchecked(
            client,
            ruda_count,
            ruda_dim,
            address_type,
            linear_view(scale),
            linear_view(out_scale),
            [input_dtype.into(), output_dtype.into()],
        )
    };
    Ok(())
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    scale: TensorBinding<R>,
    out_scale: TensorBinding<R>,
    scheme: &QuantScheme,
    input_elem: ElemType,
) -> Result<(), LaunchError> {
    launch_ref_with_scale_dtype(
        client, input, output, scale, out_scale, scheme, input_elem, input_elem,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_ref_with_scale_dtype<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    scale: TensorBinding<R>,
    out_scale: TensorBinding<R>,
    scheme: &QuantScheme,
    input_elem: ElemType,
    input_scale_elem: ElemType,
) -> Result<(), LaunchError> {
    let param_elem = ElemType::from_quant_param(scheme.param);

    match scheme {
        QuantScheme {
            store: QuantStore::PackedU32(_),
            ..
        } => quantize_packed(
            client, input, scheme, scale, out_scale, output, input_elem, input_scale_elem, param_elem,
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

            quantize_native(
                client, input, scheme, scale, out_scale, output, input_elem, input_scale_elem, param_elem,
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

#[allow(clippy::too_many_arguments)]
fn quantize_native<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    scheme: &QuantScheme,
    scale: TensorBinding<R>,
    out_scale: TensorBinding<R>,
    output: TensorBinding<R>,
    input_dtype: ElemType,
    input_scale_dtype: ElemType,
    scale_dtype: ElemType,
) -> Result<(), LaunchError> {
    let logical_shape = input.shape.clone();
    let num_elems: usize = input.shape.iter().product();
    if num_elems == 0 {
        return copy_empty_scales(client, scale, out_scale, input_scale_dtype, scale_dtype);
    }
    let (range_min, range_max) = scheme.value.range();

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            value: QuantValue::E2M1,
            store: QuantStore::PackedNative(packed_dim),
            ..
        } => {
            let packed_axis = input.shape.len() - 1 - *packed_dim;
            let axis_len = input.shape[packed_axis];
            let inner = input.shape[packed_axis + 1..].iter().product::<usize>();
            let address_type = input
                .required_address_type(input_dtype.size())
                .max(scale.required_address_type(input_scale_dtype.size()))
                .max(out_scale.required_address_type(scale_dtype.size()))
                .max(output.required_address_type(size_of::<e2m1x2>()));

            let can_vectorize = packed_axis == input.shape.len() - 1
                && tensor_vector_size_parallel(
                    core::iter::once(2),
                    &input.shape,
                    &input.strides,
                    packed_axis,
                ) == 2;
            if can_vectorize {
                let fp4_working_units = num_elems / 2;
                let fp4_ruda_dim = RudaDim::new(client.properties(), fp4_working_units);
                let fp4_ruda_count =
                    calculate_ruda_count_elemwise(client, fp4_working_units, fp4_ruda_dim);
                let layout = scales_layout(&logical_shape, &scale, 1, scheme);
                unsafe {
                    quantize_symmetric_fp4_native_kernel::launch_unchecked(
                        client,
                        fp4_ruda_count,
                        fp4_ruda_dim,
                        address_type,
                        2,
                        1,
                        linear_view(input.clone()),
                        scales_view(&logical_shape, scale.clone(), 1, scheme),
                        InputScalar::new(range_min, input_dtype),
                        InputScalar::new(range_max, input_dtype),
                        linear_view(output.clone()),
                        scales_view(&logical_shape, out_scale, 1, scheme),
                        layout,
                        [input_dtype.into(), input_scale_dtype.into(), scale_dtype.into()],
                    )
                };
            } else {
                let working_units = output.shape.iter().product();
                let ruda_dim = RudaDim::new(client.properties(), working_units);
                let ruda_count =
                    calculate_ruda_count_elemwise(client, working_units, ruda_dim);
                let layout = scales_layout(&logical_shape, &scale, 1, scheme);
                unsafe {
                    quantize_symmetric_fp4_native_strided_kernel::launch_unchecked(
                        client,
                        ruda_count,
                        ruda_dim,
                        address_type,
                        1,
                        1,
                        linear_view(input),
                        scales_view(&logical_shape, scale, 1, scheme),
                        InputScalar::new(range_min, input_dtype),
                        InputScalar::new(range_max, input_dtype),
                        linear_view(output.clone()),
                        scales_view(&logical_shape, out_scale, 1, scheme),
                        layout,
                        inner,
                        axis_len,
                        [input_dtype.into(), input_scale_dtype.into(), scale_dtype.into()],
                    )
                };
            }
        }
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            store: QuantStore::Native,
            ..
        } => {
            let quant_type = ElemType::from_quant_value(scheme.value);
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
                .required_address_type(input_dtype.size())
                .max(scale.required_address_type(input_scale_dtype.size()))
                .max(out_scale.required_address_type(scale_dtype.size()))
                .max(output.required_address_type(quant_type.size()));

            let scales_layout = scales_layout(&logical_shape, &scale, 1, scheme);

            unsafe {
                quantize_symmetric_native_kernel::launch_unchecked(
                    client,
                    ruda_count,
                    ruda_dim,
                    address_type,
                    vector_size,
                    linear_view(input),
                    scales_view(&logical_shape, scale, 1, scheme),
                    InputScalar::new(range_min, input_dtype),
                    InputScalar::new(range_max, input_dtype),
                    linear_view(output.clone()),
                    scales_view(&logical_shape, out_scale, 1, scheme),
                    scales_layout,
                    scheme.value,
                    [input_dtype.into(), input_scale_dtype.into(), scale_dtype.into(), quant_type.into()],
                )
            }
        }
        _ => panic!("Unsupported quantization scheme {scheme:?}"),
    };

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn quantize_packed<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    scheme: &QuantScheme,
    scale: TensorBinding<R>,
    out_scale: TensorBinding<R>,
    output: TensorBinding<R>,
    dtype_input: ElemType,
    dtype_input_scale: ElemType,
    dtype_param: ElemType,
) -> Result<(), LaunchError> {
    let logical_shape = input.shape.clone();
    let num_elems: usize = input.shape.iter().product();

    let packed_axis = match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            store: QuantStore::PackedU32(dim),
            ..
        } => {
            input.shape.len() - 1 - *dim
        }
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    };
    if num_elems == 0 {
        return copy_empty_scales(client, scale, out_scale, dtype_input_scale, dtype_param);
    }
    let num_quants = scheme.num_quants();
    let axis_len = input.shape[packed_axis];
    let inner = input.shape[packed_axis + 1..].iter().product::<usize>();
    let aligned_last_axis = packed_axis == input.shape.len() - 1
        && axis_len.is_multiple_of(num_quants);
    let can_vectorize = aligned_last_axis && tensor_vector_size_parallel(
        core::iter::once(num_quants),
        &input.shape,
        &input.strides,
        packed_axis,
    ) == num_quants;
    let input = if aligned_last_axis && !can_vectorize && num_elems >= 2048 {
        into_contiguous(client, input, dtype_input.into()).binding()
    } else {
        input
    };

    let vector_size = if aligned_last_axis {
        tensor_vector_size_parallel(
            core::iter::once(num_quants),
            &input.shape,
            &input.strides,
            packed_axis,
        )
    } else {
        1
    };

    let working_units = output.shape.iter().product();
    let ruda_dim = RudaDim::new(client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(client, working_units, ruda_dim);
    let (range_min, range_max) = scheme.value.range();

    let address_type = input
        .required_address_type(dtype_input.size())
        .max(scale.required_address_type(dtype_input_scale.size()))
        .max(out_scale.required_address_type(dtype_param.size()))
        .max(output.required_address_type(size_of::<u32>()));

    let scales_layout = scales_layout(&logical_shape, &scale, 1, scheme);

    unsafe {
        quantize_symmetric_packed_kernel::launch_unchecked(
            client,
            ruda_count,
            ruda_dim,
            address_type,
            vector_size,
            linear_view(input),
            scales_view(&logical_shape, scale, 1, scheme),
            InputScalar::new(range_min, dtype_input),
            InputScalar::new(range_max, dtype_input),
            linear_view(output.clone()),
            scales_view(&logical_shape, out_scale, 1, scheme),
            scales_layout,
            inner,
            axis_len,
            *scheme,
            [dtype_input.into(), dtype_input_scale.into(), dtype_param.into()],
        )
    };

    Ok(())
}
