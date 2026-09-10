use super::{allocate_output, packing_layout, raw_dtype, read_value};
use ruda_core::tensor::{QTensorPrimitive, QuantScheme, TensorMetadata};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{Runtime, calculate_ruda_count_elemwise, prelude::*};
use ruda_kernel::library::{FastDivmod, tensor::layout::linear::LinearView};
use ruda_kernel::tensor::{RudaTensor, layout::{address_type, shape_divmod}};

pub fn quantized_gather<R: Runtime>(
    dim: usize,
    tensor: RudaTensor<R>,
    indices: RudaTensor<R>,
) -> RudaTensor<R> {
    assert_eq!(tensor.rank(), indices.rank(), "Gather rank mismatch");
    assert!(dim < tensor.rank(), "Gather dimension out of bounds");
    for axis in 0..tensor.rank() {
        if axis != dim {
            assert_eq!(tensor.meta.shape()[axis], indices.meta.shape()[axis], "Gather shape mismatch");
        }
    }
    let output = allocate_output(&tensor, indices.shape());
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
        gather_kernel::launch_unchecked(
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

#[ruda(launch_unchecked, address_type = "dynamic")]
fn gather_kernel<T: Int, I: Numeric>(
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
    let group = ABSOLUTE_POS / inner;
    let word = group % axis_words;
    let outer = group / axis_words;
    let inner_pos = ABSOLUTE_POS % inner;
    let base = (outer * axis_len + word * packing) * inner + inner_pos;
    let mut packed = T::new(0);
    #[unroll]
    for lane in 0..packing {
        if word * packing + lane < axis_len {
            let selected = usize::cast_from(indices[base + lane * inner]);
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
                    coord = selected;
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
