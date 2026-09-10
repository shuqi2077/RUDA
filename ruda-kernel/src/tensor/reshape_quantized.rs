use super::{RudaTensor, allocation::empty_qtensor_optimized, layout::{address_type, shape_divmod}};
use crate::dsl::{Runtime, calculate_ruda_count_elemwise, prelude::*};
use crate::library::{FastDivmod, tensor::layout::linear::LinearView};
use ruda_core::tensor::{
    DType, Metadata, QTensorPrimitive, QuantLevel, QuantScheme, QuantStore, ReshapeAction, Shape,
    TensorMetadata, contiguous_strides, params_shape, reshape_action,
};

pub(super) fn try_reshape<R: Runtime>(
    mut tensor: RudaTensor<R>,
    shape: Shape,
) -> Result<RudaTensor<R>, (RudaTensor<R>, Shape)> {
    assert_eq!(tensor.meta.num_elements(), shape.num_elements(), "Reshape element count mismatch");
    let scheme = *tensor.scheme();
    if tensor.meta.shape() == &shape {
        return Ok(tensor);
    }
    let packing = scheme.num_quants();
    let (in_axis, out_axis) = match scheme.store {
        QuantStore::Native => (tensor.rank().saturating_sub(1), shape.rank().saturating_sub(1)),
        QuantStore::PackedU32(dim) | QuantStore::PackedNative(dim) => {
            assert!(shape.rank() > dim, "Reshape target has no axis for the quantization packing dimension");
            (tensor.rank() - dim - 1, shape.rank() - dim - 1)
        }
    };
    let mut value_shape = shape.clone();
    if packing > 1 {
        value_shape[out_axis] = value_shape[out_axis].div_ceil(packing);
    }
    let same_suffix = tensor.meta.shape().iter().skip_while(|dim| **dim == 1)
        .eq(shape.iter().skip_while(|dim| **dim == 1));
    if shape.num_elements() != 0
        && !scale_mapping_preserved(tensor.meta.shape(), &shape, scheme.level, same_suffix)
    {
        return Err((tensor, shape));
    }
    let shape_scales = params_shape(&shape, scheme.level);
    let (values, scales) = tensor.quantized_handles().unwrap();
    if shape.num_elements() == 0 {
        *tensor.meta = Metadata::new(shape, contiguous_strides(&value_shape));
        if matches!(scheme.level, QuantLevel::Block(_)) {
            let strides = contiguous_strides(&shape_scales);
            tensor.qparams.as_mut().unwrap().scales.metadata = Metadata::new(shape_scales, strides);
        }
        return Ok(tensor);
    }
    let packed_linear = packing == 1 || same_suffix || (
        in_axis + 1 == tensor.rank() && out_axis + 1 == shape.rank()
        && tensor.meta.shape()[in_axis].is_multiple_of(packing)
        && shape[out_axis].is_multiple_of(packing)
    );
    if packed_linear {
        let value_strides = match reshape_action(values.meta.shape(), values.meta.strides(), &value_shape) {
            ReshapeAction::NoChange => Some(values.meta.strides().clone()),
            ReshapeAction::UpdateStrides { strides } => Some(strides),
            ReshapeAction::Recompute => None,
        };
        let scale_strides = match reshape_action(scales.meta.shape(), scales.meta.strides(), &shape_scales) {
            ReshapeAction::NoChange => Some(scales.meta.strides().clone()),
            ReshapeAction::UpdateStrides { strides } => Some(strides),
            ReshapeAction::Recompute => None,
        };
        if let (Some(value_strides), Some(scale_strides)) = (value_strides, scale_strides) {
            *tensor.meta = Metadata::new(shape, value_strides);
            tensor.qparams.as_mut().unwrap().scales.metadata = Metadata::new(shape_scales, scale_strides);
            return Ok(tensor);
        }
    }

    let output = empty_qtensor_optimized(shape.clone(), scheme, &tensor.device);
    let (out_values, out_scales) = output.quantized_handles().unwrap();
    let scale_dtype = scales.dtype;
    crate::library::tensor::copy_into(
        &output.client,
        scales.binding(),
        out_scales.binding(),
        scale_dtype.into(),
    );
    let num_elems = out_values.meta.num_elements();
    if num_elems == 0 {
        return Ok(output);
    }

    let ruda_dim = RudaDim::new(output.client.properties(), num_elems);
    let ruda_count = calculate_ruda_count_elemwise(&output.client, num_elems, ruda_dim);
    let dtype = match values.dtype {
        DType::I8 => DType::U8,
        other => other,
    };
    let inner = shape.iter().skip(out_axis + 1).product::<usize>();
    let axis_len = shape.get(out_axis).copied().unwrap_or(1);
    unsafe {
        reshape_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            address_type!(values, out_values)
                .max(AddressType::from_len(shape.num_elements())),
            values.into_tensor_arg(),
            out_values.into_linear_view(),
            shape_divmod(&tensor),
            inner,
            axis_len,
            in_axis,
            scheme,
            dtype.into(),
        );
    }
    Ok(output)
}

fn scale_mapping_preserved(input: &Shape, output: &Shape, level: QuantLevel, same_suffix: bool) -> bool {
    let QuantLevel::Block(block) = level else {
        return true;
    };
    let input_blocks = block.to_dim_vec(input.rank());
    let output_blocks = block.to_dim_vec(output.rank());
    assert!(!input_blocks.contains(&0) && !output_blocks.contains(&0), "Quantization block dimensions must be nonzero");
    if same_suffix {
        let input_leading = input.iter().take_while(|dim| **dim == 1).count();
        let output_leading = output.iter().take_while(|dim| **dim == 1).count();
        if input_blocks[input_leading..] == output_blocks[output_leading..] {
            return true;
        }
    }
    match (linear_block_size(input, &input_blocks), linear_block_size(output, &output_blocks)) {
        (Some(input), Some(output)) => input == output,
        _ => false,
    }
}

fn linear_block_size(shape: &Shape, blocks: &[u8]) -> Option<usize> {
    let mut inner_complete = true;
    let mut elements = 1;
    for (&dim, &block) in shape.iter().zip(blocks).rev() {
        let block = block as usize;
        if !dim.is_multiple_of(block) || (block > 1 && !inner_complete) {
            return None;
        }
        inner_complete &= dim == block;
        elements *= block;
    }
    Some(elements)
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn reshape_kernel<T: Int>(
    input: &Tensor<T>,
    output: &mut LinearView<T, ReadWrite>,
    in_shape: Sequence<FastDivmod<usize>>,
    inner: usize,
    axis_len: usize,
    #[comptime] input_packed_axis: usize,
    #[comptime] scheme: QuantScheme,
    #[define(T)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    let rank = in_shape.len().comptime();
    let packing = scheme.num_quants();
    let bits = scheme.value.size_bits();
    let mask = T::cast_from((1u32 << bits) - 1);
    let axis_words = axis_len / packing + usize::cast_from(axis_len % packing != 0);
    let group = ABSOLUTE_POS / inner;
    let word = group % axis_words;
    let outer = group / axis_words;
    let inner_pos = ABSOLUTE_POS % inner;
    let base = (outer * axis_len + word * packing) * inner + inner_pos;
    let mut packed = T::new(0);
    #[unroll]
    for lane in 0..packing {
        if word * packing + lane < axis_len {
            let mut remainder = base + lane * inner;
            let mut offset = 0;
            let mut slot = 0;
            #[unroll]
            for i in 0..rank {
                let axis = rank - i - 1;
                let (rem, mut coord) = in_shape[axis].div_mod(remainder);
                remainder = rem;
                if axis == input_packed_axis {
                    slot = coord % packing;
                    coord /= packing;
                }
                offset += coord * input.stride(axis);
            }
            let value = (input[offset] >> T::cast_from(slot * bits)) & mask;
            packed |= value << T::cast_from(lane * bits);
        }
    }
    output[ABSOLUTE_POS] = packed;
}
