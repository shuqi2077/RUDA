use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{Runtime, calculate_ruda_count_elemwise, num_traits::Zero, prelude::*};
use ruda_kernel::library::FastDivmod;
use ruda_kernel::tensor::{
    RudaTensor,
    allocation::empty_device_dtype,
    contiguous::into_contiguous_aligned,
    layout::{address_type, decompose_linear, max_vector_size, shape_divmod},
    permutation::{permute_nchw_to_nhwc, permute_nhwc_to_nchw},
};
use ruda_core::{ir::AddressType, tensor::{DType, Shape}};
use crate::indexing::scaled_index_division;

#[ruda]
fn bin_start(index: usize, output_extent: usize, input_extent: usize,
    #[comptime] max_value: usize) -> usize {
    let (quotient, _) = scaled_index_division(index, input_extent, output_extent, max_value);
    quotient
}

#[ruda]
fn bin_end(index: usize, output_extent: usize, input_extent: usize,
    #[comptime] max_value: usize) -> usize {
    let (quotient, remainder) = scaled_index_division(index + 1, input_extent, output_extent, max_value);
    if remainder == 0 { quotient } else { quotient + 1 }
}

#[ruda(launch, address_type = "dynamic")]
fn adaptive_average_volume<E: Numeric, A: Float, N: Size>(
    input: &Tensor<Vector<E, N>>,
    output: &mut Tensor<Vector<E, N>>,
    output_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[comptime] max_value: usize,
    #[define(E)] _storage: StorageType,
    #[define(A)] _compute: StorageType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }
    let (_, position) = decompose_linear(ABSOLUTE_POS * output.vector_size(), &output_shape);
    let [batch, od, oh, ow, channel] = *position else { unreachable!() };
    let id_start = bin_start(od, output.shape(1), input.shape(1), max_value);
    let id_end = bin_end(od, output.shape(1), input.shape(1), max_value);
    let ih_start = bin_start(oh, output.shape(2), input.shape(2), max_value);
    let ih_end = bin_end(oh, output.shape(2), input.shape(2), max_value);
    let iw_start = bin_start(ow, output.shape(3), input.shape(3), max_value);
    let iw_end = bin_end(ow, output.shape(3), input.shape(3), max_value);
    let base = batch * input.stride(0) + channel * input.stride(4);
    let mut sum = Vector::<A, N>::zero();
    for id in id_start..id_end {
        for ih in ih_start..ih_end {
            for iw in iw_start..iw_end {
                let index = base + id * input.stride(1) + ih * input.stride(2) + iw * input.stride(3);
                sum += Vector::cast_from(input[index / input.vector_size()]);
            }
        }
    }
    let count = (id_end - id_start) * (ih_end - ih_start) * (iw_end - iw_start);
    output[ABSOLUTE_POS] = Vector::cast_from(sum / Vector::cast_from(count));
}

#[ruda(launch, address_type = "dynamic")]
fn adaptive_average_volume_backward<E: Numeric, G: Numeric, A: Float, N: Size>(
    grad: &Tensor<Vector<G, N>>,
    output: &mut Tensor<Vector<E, N>>,
    output_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[comptime] max_value: usize,
    #[define(E)] _storage: StorageType,
    #[define(G)] _gradient_storage: StorageType,
    #[define(A)] _compute: StorageType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }
    let (_, position) = decompose_linear(ABSOLUTE_POS * output.vector_size(), &output_shape);
    let [batch, id, ih, iw, channel] = *position else { unreachable!() };
    let od_start = bin_start(id, output.shape(1), grad.shape(1), max_value);
    let od_end = bin_end(id, output.shape(1), grad.shape(1), max_value);
    let oh_start = bin_start(ih, output.shape(2), grad.shape(2), max_value);
    let oh_end = bin_end(ih, output.shape(2), grad.shape(2), max_value);
    let ow_start = bin_start(iw, output.shape(3), grad.shape(3), max_value);
    let ow_end = bin_end(iw, output.shape(3), grad.shape(3), max_value);
    let base = batch * grad.stride(0) + channel * grad.stride(4);
    let mut sum = Vector::<A, N>::zero();
    for od in od_start..od_end {
        let id_start = bin_start(od, grad.shape(1), output.shape(1), max_value);
        let id_end = bin_end(od, grad.shape(1), output.shape(1), max_value);
        if id >= id_start && id < id_end {
            for oh in oh_start..oh_end {
                let ih_start = bin_start(oh, grad.shape(2), output.shape(2), max_value);
                let ih_end = bin_end(oh, grad.shape(2), output.shape(2), max_value);
                if ih >= ih_start && ih < ih_end {
                    for ow in ow_start..ow_end {
                        let iw_start = bin_start(ow, grad.shape(3), output.shape(3), max_value);
                        let iw_end = bin_end(ow, grad.shape(3), output.shape(3), max_value);
                        if iw >= iw_start && iw < iw_end {
                            let count = (id_end - id_start) * (ih_end - ih_start) * (iw_end - iw_start);
                            let index = base + od * grad.stride(1) + oh * grad.stride(2) + ow * grad.stride(3);
                            let value = Vector::<A, N>::cast_from(grad[index / grad.vector_size()]);
                            sum += value / Vector::cast_from(count);
                        }
                    }
                }
            }
        }
    }
    output[ABSOLUTE_POS] = Vector::cast_from(sum);
}

pub(super) fn accumulation_dtype(storage: DType) -> DType {
    if storage == DType::F64 { DType::F64 } else { DType::F32 }
}

fn bin_address_type<R: Runtime>(input: &RudaTensor<R>, output: &RudaTensor<R>) -> AddressType {
    let wide_bins = (1..4).any(|axis| input.meta.shape()[axis]
        .checked_mul(output.meta.shape()[axis]).is_none_or(|product| product > u32::MAX as usize));
    if wide_bins { AddressType::U64 } else { address_type!(input, output) }
}

/// Adaptive average pooling of native `[batch, channels, depth, height, width]` volumes.
///
/// Reduces each actual three-dimensional bin in one device kernel. Half-storage
/// accumulation uses FP32; F64 storage retains F64 accumulation. Output storage
/// and device remain those of the input. Non-contiguous inputs are supported.
pub fn adaptive_avg_pool3d<R: Runtime>(input: RudaTensor<R>, output_size: [usize; 3]) -> RudaTensor<R> {
    let [batch, channels, _, _, _] = input.meta.shape().dims();
    let input = into_contiguous_aligned(permute_nchw_to_nhwc(input));
    let output = empty_device_dtype(input.client.clone(), input.device.clone(),
        Shape::new([batch, output_size[0], output_size[1], output_size[2], channels]), input.dtype);
    let vector_size = max_vector_size(&input);
    let working_units = output.meta.num_elements() / vector_size as usize;
    if working_units == 0 {
        return permute_nhwc_to_nchw(output);
    }
    let dim = RudaDim::new(input.client.properties(), working_units);
    let count = calculate_ruda_count_elemwise(&input.client, working_units, dim);
    let address = bin_address_type(&input, &output);
    let max_value = if address == AddressType::U64 { usize::MAX } else { u32::MAX as usize };
    adaptive_average_volume::launch(&output.client, count, dim, address,
        vector_size, input.into_tensor_arg(), output.clone().into_tensor_arg(),
        shape_divmod(&output), working_units, max_value, output.dtype.into(), accumulation_dtype(output.dtype).into());
    permute_nhwc_to_nchw(output)
}

/// Native input gradients for three-dimensional adaptive average pooling.
///
/// Every input element gathers all bins covering it, including overlapping bins
/// when output extents exceed input extents. No floating atomic additions or
/// host reductions are used; accumulation follows the forward working dtype.
pub fn adaptive_avg_pool3d_backward<R: Runtime>(input: RudaTensor<R>, grad: RudaTensor<R>) -> RudaTensor<R> {
    let [batch, channels, depth, height, width] = input.meta.shape().dims();
    let [grad_batch, grad_channels, _, _, _] = grad.meta.shape().dims();
    assert_eq!([grad_batch, grad_channels], [batch, channels], "adaptive pooling gradient batch/channels differ");
    let grad = into_contiguous_aligned(permute_nchw_to_nhwc(grad));
    let output = empty_device_dtype(input.client.clone(), input.device.clone(),
        Shape::new([batch, depth, height, width, channels]), input.dtype);
    let vector_size = max_vector_size(&grad);
    let working_units = output.meta.num_elements() / vector_size as usize;
    if working_units == 0 {
        return permute_nhwc_to_nchw(output);
    }
    let dim = RudaDim::new(input.client.properties(), working_units);
    let count = calculate_ruda_count_elemwise(&input.client, working_units, dim);
    let address = bin_address_type(&grad, &output);
    let max_value = if address == AddressType::U64 { usize::MAX } else { u32::MAX as usize };
    let gradient_storage = grad.dtype;
    let compute = if gradient_storage == DType::F64 { DType::F64 } else { accumulation_dtype(output.dtype) };
    adaptive_average_volume_backward::launch(&output.client, count, dim, address,
        vector_size, grad.into_tensor_arg(), output.clone().into_tensor_arg(), shape_divmod(&output),
        working_units, max_value, output.dtype.into(), gradient_storage.into(), compute.into());
    permute_nhwc_to_nchw(output)
}
