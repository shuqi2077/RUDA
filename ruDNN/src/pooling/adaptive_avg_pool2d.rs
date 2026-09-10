use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::contiguous::into_contiguous_aligned;
use crate::pooling::pool2d::Position;
use crate::pooling::pool2d::view4d;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::decompose_linear;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::layout::max_vector_size;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::permutation::permute_nchw_to_nhwc;
use ruda_kernel::tensor::permutation::permute_nhwc_to_nchw;
use ruda_kernel::tensor::RudaTensor;
use ruda_core::tensor::Shape;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::num_traits::Zero;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;
use ruda_kernel::library::tensor::View;

#[ruda(launch, address_type = "dynamic")]
fn adaptive_avg_pool2d_direct<E: Numeric, N: Size>(
    input: &Tensor<Vector<E, N>>,
    output: &mut View<Vector<E, N>, Position, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let (_, pos) = decompose_linear(ABSOLUTE_POS * output.vector_size(), &out_shape);
    let [b, oh, ow, c] = *pos else { unreachable!() };

    let (_, out_h, out_w, _) = output.shape();
    let (in_stride_h, in_stride_w) = (input.stride(1), input.stride(2));
    let (in_h, in_w) = (input.shape(1), input.shape(2));

    let ih_start = start_index(oh, out_h, in_h);
    let ih_end = end_index(oh, out_h, in_h);

    let iw_start = start_index(ow, out_w, in_w);
    let iw_end = end_index(ow, out_w, in_w);

    let mut sum = Vector::zero();

    let index_input_base = b * input.stride(0) + c * input.stride(3);

    for ih in ih_start..ih_end {
        let index_input_2 = ih * in_stride_h;

        for iw in iw_start..iw_end {
            let index_input_3 = iw * in_stride_w;

            let index_input = index_input_base + index_input_2 + index_input_3;
            sum += input[index_input / input.vector_size()];
        }
    }

    let num_ih = ih_end - ih_start;
    let num_iw = iw_end - iw_start;

    output[(b, oh, ow, c)] = sum / Vector::cast_from(num_ih * num_iw);
}

#[ruda]
fn start_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    (output_size_index * input_size) / output_size
}

#[ruda]
fn end_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    let index = (output_size_index + 1) * input_size;
    let index = index.div_ceil(output_size);

    if input_size < index {
        input_size
    } else {
        index
    }
}

pub fn adaptive_avg_pool2d<R: Runtime>(
    input: RudaTensor<R>,
    output_size: [usize; 2],
) -> RudaTensor<R> {
    let [batch_size, channels, _, _] = input.meta.shape().dims();

    let input = into_contiguous_aligned(permute_nchw_to_nhwc(input));
    let vector_size = max_vector_size(&input);

    let output_shape = Shape::new([batch_size, output_size[0], output_size[1], channels]);
    let num_elems: usize = output_shape.num_elements();
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        output_shape,
        input.dtype,
    );

    let working_units = num_elems / vector_size as usize;
    let ruda_dim = RudaDim::new(input.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&input.client, working_units, ruda_dim);

    adaptive_avg_pool2d_direct::launch(
        &output.client,
        ruda_count,
        ruda_dim,
        address_type!(input, output),
        vector_size,
        input.into_tensor_arg(),
        view4d(output.clone(), vector_size),
        shape_divmod(&output),
        working_units,
        output.dtype.into(),
    );

    permute_nhwc_to_nchw(output)
}
