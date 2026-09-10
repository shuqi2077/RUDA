use super::{RudaTensor, allocation::empty_qtensor_optimized, layout::address_type};
use crate::dsl::{Runtime, calculate_ruda_count_elemwise, prelude::*};
use crate::library::tensor::layout::linear::LinearView;
use ruda_core::tensor::{Metadata, QTensorPrimitive, Shape, TensorMetadata};
use ruda_core::tensor::quantization::{QuantStore, params_shape};

pub(super) fn expand<R: Runtime>(mut tensor: RudaTensor<R>, shape: Shape) -> RudaTensor<R> {
    let rank = tensor.rank();
    let target_rank = shape.num_dims();
    assert!(target_rank >= rank, "Cannot expand a tensor to a lower rank");
    let leading = target_rank - rank;
    for axis in 0..rank {
        let source = tensor.meta.shape()[axis];
        assert!(
            source == shape[leading + axis] || source == 1,
            "Cannot expand dimension {source} to {}",
            shape[leading + axis],
        );
    }

    let scheme = *tensor.scheme();
    let mut value_shape = shape.clone();
    let mut broadcast_packed_axis = None;
    match scheme.store {
        QuantStore::Native => {}
        QuantStore::PackedU32(dim) | QuantStore::PackedNative(dim) => {
            let axis = target_rank - dim - 1;
            value_shape[axis] = shape[axis].div_ceil(scheme.num_quants());
            if tensor.meta.shape()[rank - dim - 1] == 1 && shape[axis] > 1 {
                broadcast_packed_axis = Some(axis);
            }
        }
    }

    let (values, scales) = tensor.quantized_handles().unwrap();
    let values = super::view::expand(values, value_shape);
    let scales = super::view::expand(scales, params_shape(&shape, scheme.level));

    if let Some(axis) = broadcast_packed_axis {
        let output = empty_qtensor_optimized(shape.clone(), scheme, &tensor.device);
        let (out_values, out_scales) = output.quantized_handles().unwrap();
        let num_elems = out_values.meta.num_elements();
        let client = output.client.clone();
        if num_elems > 0 {
            let ruda_dim = RudaDim::new(client.properties(), num_elems);
            let ruda_count = calculate_ruda_count_elemwise(&client, num_elems, ruda_dim);
            let dtype = values.dtype;
            let inner = shape[axis + 1..].iter().product::<usize>();
            unsafe {
                repeat_packed::launch_unchecked(
                    &client,
                    ruda_count,
                    ruda_dim,
                    address_type!(values, out_values)
                        .max(AddressType::from_len(shape.num_elements())),
                    values.into_linear_view(),
                    out_values.into_linear_view(),
                    inner,
                    shape[axis],
                    scheme.num_quants(),
                    scheme.value.size_bits(),
                    dtype.into(),
                );
            }
        }
        let scale_dtype = scales.dtype;
        crate::library::tensor::copy_into(
            &client,
            scales.binding(),
            out_scales.binding(),
            scale_dtype.into(),
        );
        output
    } else {
        *tensor.meta = Metadata::new(shape, values.meta.strides().clone());
        tensor.qparams.as_mut().unwrap().scales.metadata = *scales.meta;
        tensor
    }
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn repeat_packed<T: Int>(
    input: &LinearView<T>,
    output: &mut LinearView<T, ReadWrite>,
    inner: usize,
    axis_len: usize,
    #[comptime] packing: usize,
    #[comptime] bits: usize,
    #[define(T)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    let packed_axis_len = axis_len / packing + usize::cast_from(axis_len % packing != 0);
    let axis_pos = (ABSOLUTE_POS / inner) % packed_axis_len;
    let mask = T::cast_from((1u32 << bits) - 1);
    let value = input[ABSOLUTE_POS] & mask;
    let mut packed = T::new(0);
    #[unroll]
    for lane in 0..packing {
        if axis_pos * packing + lane < axis_len {
            packed |= value << T::cast_from(lane * bits);
        }
    }
    output[ABSOLUTE_POS] = packed;
}
