use ruda_core::tensor::{
    DType, QTensorPrimitive, QuantLevel, QuantScheme, QuantStore, Shape, Slice, SliceOps,
    TensorMetadata,
};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{Runtime, calculate_ruda_count_elemwise, prelude::*};
use ruda_kernel::library::{FastDivmod, tensor::layout::linear::LinearView};
use ruda_kernel::tensor::{
    RudaTensor, allocation::empty_qtensor_optimized, layout::{address_type, shape_divmod},
};

mod gather;
pub use gather::quantized_gather;

pub fn quantized_flip<R: Runtime>(tensor: RudaTensor<R>, axes: &[usize]) -> RudaTensor<R> {
    let slices = (0..tensor.rank())
        .map(|axis| Slice::new(0, None, if axes.contains(&axis) { -1 } else { 1 }))
        .collect::<Vec<_>>();
    quantized_slice(tensor, &slices)
}

pub fn quantized_slice<R: Runtime>(tensor: RudaTensor<R>, slices: &[Slice]) -> RudaTensor<R> {
    let shape = tensor.shape().slice(slices).unwrap();
    let output = allocate_output(&tensor, shape);
    let (values, _) = tensor.quantized_handles().unwrap();
    let (out_values, _) = output.quantized_handles().unwrap();
    let num_elems = out_values.meta.num_elements();
    if num_elems == 0 {
        return output;
    }

    let mut origins = SequenceArg::<R, usize>::new();
    let mut steps = SequenceArg::<R, usize>::new();
    let mut reversed = SequenceArg::<R, usize>::new();
    for axis in 0..tensor.rank() {
        let slice = slices.get(axis).copied().unwrap_or_default();
        let range = slice.to_range(tensor.meta.shape()[axis]);
        origins.push(if slice.is_reversed() { range.end - 1 } else { range.start });
        steps.push(slice.step.unsigned_abs());
        reversed.push(usize::from(slice.is_reversed()));
    }

    let (axis, inner, axis_len) = packing_layout(&output);
    let ruda_dim = RudaDim::new(output.client.properties(), num_elems);
    let ruda_count = calculate_ruda_count_elemwise(&output.client, num_elems, ruda_dim);
    let dtype = raw_dtype(values.dtype);
    let step_address = AddressType::from_len(
        slices.iter().map(|slice| slice.step.unsigned_abs()).max().unwrap_or(1),
    );
    unsafe {
        affine_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            address_type!(values, out_values).max(AddressType::from_len(
                tensor.meta.num_elements().max(output.meta.num_elements()),
            )).max(step_address),
            values.into_tensor_arg(),
            out_values.clone().into_linear_view(),
            shape_divmod(&out_values),
            origins,
            steps,
            reversed,
            inner,
            axis_len,
            axis,
            *tensor.scheme(),
            dtype.into(),
        );
    }
    output
}

pub fn quantized_select<R: Runtime>(
    tensor: RudaTensor<R>,
    dim: usize,
    indices: RudaTensor<R>,
) -> RudaTensor<R> {
    let mut shape = tensor.shape();
    shape[dim] = indices.meta.shape()[0];
    let output = allocate_output(&tensor, shape);
    let (values, _) = tensor.quantized_handles().unwrap();
    let (out_values, _) = output.quantized_handles().unwrap();
    let num_elems = out_values.meta.num_elements();
    if num_elems == 0 {
        return output;
    }

    let (axis, inner, axis_len) = packing_layout(&output);
    let ruda_dim = RudaDim::new(output.client.properties(), num_elems);
    let ruda_count = calculate_ruda_count_elemwise(&output.client, num_elems, ruda_dim);
    let dtypes = [raw_dtype(values.dtype).into(), indices.dtype.into()];
    unsafe {
        select_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            address_type!(values, indices, out_values).max(AddressType::from_len(
                tensor.meta.num_elements().max(output.meta.num_elements()),
            )),
            values.into_tensor_arg(),
            indices.into_linear_view(),
            out_values.clone().into_linear_view(),
            shape_divmod(&out_values),
            inner,
            axis_len,
            axis,
            dim,
            *tensor.scheme(),
            dtypes,
        );
    }
    output
}

fn allocate_output<R: Runtime>(tensor: &RudaTensor<R>, shape: Shape) -> RudaTensor<R> {
    assert_eq!(tensor.scheme().level, QuantLevel::Tensor);
    let output = empty_qtensor_optimized(shape, *tensor.scheme(), &tensor.device);
    let scales = tensor.scales().unwrap();
    let out_scales = output.scales().unwrap();
    let dtype = scales.dtype;
    ruda_kernel::library::tensor::copy_into(
        &output.client,
        scales.binding(),
        out_scales.binding(),
        dtype.into(),
    );
    output
}

fn raw_dtype(dtype: DType) -> DType {
    match dtype {
        DType::I8 => DType::U8,
        other => other,
    }
}

fn packing_layout<R: Runtime>(tensor: &RudaTensor<R>) -> (usize, usize, usize) {
    let axis = match tensor.scheme().store {
        QuantStore::Native => tensor.rank().saturating_sub(1),
        QuantStore::PackedU32(dim) | QuantStore::PackedNative(dim) => tensor.rank() - dim - 1,
    };
    let shape = tensor.meta.shape();
    let inner = shape.iter().skip(axis + 1).product();
    let axis_len = shape.get(axis).copied().unwrap_or(1);
    (axis, inner, axis_len)
}

#[ruda]
fn read_value<T: Int>(
    input: &Tensor<T>,
    offset: usize,
    slot: usize,
    #[comptime] scheme: QuantScheme,
) -> T {
    let bits = scheme.value.size_bits();
    let mask = T::cast_from((1u32 << bits) - 1);
    (input[offset] >> T::cast_from(slot * bits)) & mask
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn affine_kernel<T: Int>(
    input: &Tensor<T>,
    output: &mut LinearView<T, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    origins: Sequence<usize>,
    steps: Sequence<usize>,
    reversed: Sequence<usize>,
    inner: usize,
    axis_len: usize,
    #[comptime] packed_axis: usize,
    #[comptime] scheme: QuantScheme,
    #[define(T)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    let rank = out_shape.len().comptime();
    let packing = scheme.num_quants();
    let bits = scheme.value.size_bits();
    let axis_words = axis_len / packing + usize::cast_from(axis_len % packing != 0);
    let word = (ABSOLUTE_POS / inner) % axis_words;
    let mut packed = T::new(0);
    #[unroll]
    for lane in 0..packing {
        if word * packing + lane < axis_len {
            let mut remainder = ABSOLUTE_POS;
            let mut offset = 0;
            let mut slot = 0;
            #[unroll]
            for i in 0..rank {
                let axis = rank - i - 1;
                let (rem, mut coord) = out_shape[axis].div_mod(remainder);
                remainder = rem;
                if axis == packed_axis {
                    coord = coord * packing + lane;
                }
                coord = if reversed[axis] != 0 {
                    origins[axis] - coord * steps[axis]
                } else {
                    origins[axis] + coord * steps[axis]
                };
                if axis == packed_axis {
                    slot = coord % packing;
                    coord /= packing;
                }
                offset += coord * input.stride(axis);
            }
            let value = read_value(input, offset, slot, scheme);
            packed |= value << T::cast_from(lane * bits);
        }
    }
    output[ABSOLUTE_POS] = packed;
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn select_kernel<T: Int, I: Numeric>(
    input: &Tensor<T>,
    indices: &LinearView<I>,
    output: &mut LinearView<T, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    inner: usize,
    axis_len: usize,
    #[comptime] packed_axis: usize,
    #[comptime] selected_axis: usize,
    #[comptime] scheme: QuantScheme,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    let rank = out_shape.len().comptime();
    let packing = scheme.num_quants();
    let bits = scheme.value.size_bits();
    let axis_words = axis_len / packing + usize::cast_from(axis_len % packing != 0);
    let word = (ABSOLUTE_POS / inner) % axis_words;
    let mut packed = T::new(0);
    #[unroll]
    for lane in 0..packing {
        if word * packing + lane < axis_len {
            let mut remainder = ABSOLUTE_POS;
            let mut offset = 0;
            let mut slot = 0;
            #[unroll]
            for i in 0..rank {
                let axis = rank - i - 1;
                let (rem, mut coord) = out_shape[axis].div_mod(remainder);
                remainder = rem;
                if axis == packed_axis {
                    coord = coord * packing + lane;
                }
                if axis == selected_axis {
                    coord = usize::cast_from(indices[coord]);
                }
                if axis == packed_axis {
                    slot = coord % packing;
                    coord /= packing;
                }
                offset += coord * input.stride(axis);
            }
            let value = read_value(input, offset, slot, scheme);
            packed |= value << T::cast_from(lane * bits);
        }
    }
    output[ABSOLUTE_POS] = packed;
}
