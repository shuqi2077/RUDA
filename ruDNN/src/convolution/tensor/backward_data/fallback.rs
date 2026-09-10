use {ruda_core::tensor::TensorMetadata, ruda_core::tensor::spatial::ConvOptions, ruda_core::tensor::spatial::ConvTransposeOptions, ruda_core::tensor::spatial::calculate_padding_out};
use {ruda_core::tensor::Shape};
use {crate::convolution::components::ConvSetupError};

use {ruda_kernel::dsl::Runtime, crate::convolution::tensor::conv_transpose2d, crate::convolution::tensor::conv_transpose3d, ruda_kernel::tensor::permutation::permute_nchw_to_nhwc, ruda_kernel::tensor::permutation::permute_nhwc_to_nchw, ruda_kernel::tensor::reshape::reshape, ruda_kernel::tensor::RudaTensor};

pub(crate) fn conv_data_backward_fallback<R: Runtime, const N_DIM: usize>(
    out_grad: RudaTensor<R>,
    weights: RudaTensor<R>,
    in_shape: Shape,
    options: ConvOptions<N_DIM>,
) -> Result<RudaTensor<R>, ConvSetupError> {
    let dim_c = out_grad.rank();

    let kernel_size = &weights.meta.shape()[1..dim_c];
    let in_shape = &in_shape[1..dim_c];
    let out_shape = &out_grad.meta.shape()[1..dim_c];

    let mut padding_out = [0; N_DIM];

    for i in 0..N_DIM {
        padding_out[i] = calculate_padding_out(
            kernel_size[i],
            options.stride[i],
            options.padding[i],
            options.dilation[i],
            in_shape[i],
            out_shape[i],
        );
    }

    // We don't yet have NHWC kernels for conv_transpose so need to do this.
    // Should eventually use NHWC kernels instead
    let out_grad = permute_nhwc_to_nchw(out_grad);
    let weights = permute_nhwc_to_nchw(weights);

    let in_grad = match N_DIM {
        1 => conv_transpose1d_from_conv_transpose2d(
            out_grad,
            weights,
            ConvTransposeOptions::new(
                [options.stride[0]],
                [options.padding[0]],
                [padding_out[0]],
                [options.dilation[0]],
                options.groups,
            ),
        ),
        2 => conv_transpose2d(
            out_grad,
            weights,
            None,
            ConvTransposeOptions::new(
                [options.stride[0], options.stride[1]],
                [options.padding[0], options.padding[1]],
                [padding_out[0], padding_out[1]],
                [options.dilation[0], options.dilation[1]],
                options.groups,
            ),
            Default::default(),
        ),
        3 => Ok(conv_transpose3d(
            out_grad,
            weights,
            None,
            ConvTransposeOptions::new(
                [options.stride[0], options.stride[1], options.stride[2]],
                [options.padding[0], options.padding[1], options.padding[2]],
                [padding_out[0], padding_out[1], padding_out[2]],
                [
                    options.dilation[0],
                    options.dilation[1],
                    options.dilation[2],
                ],
                options.groups,
            ),
        )
        .unwrap()),
        _ => unimplemented!("Invalid dimensionality"),
    }?;
    Ok(permute_nchw_to_nhwc(in_grad))
}

fn conv_transpose1d_from_conv_transpose2d<R: Runtime>(
    x: RudaTensor<R>,
    weight: RudaTensor<R>,
    options: ConvTransposeOptions<1>,
) -> Result<RudaTensor<R>, ConvSetupError> {
    let [channels_in, channels_out, kernel_size] = weight.shape().dims();
    let [batch_size, _channels_in, length_in] = x.shape().dims();

    let weight = reshape(
        weight,
        Shape::new([channels_in, channels_out, kernel_size, 1]),
    );
    let x = reshape(x, Shape::new([batch_size, channels_in, length_in, 1]));

    let tensor = conv_transpose2d(
        x,
        weight,
        None,
        ConvTransposeOptions::new(
            [options.stride[0], 1],
            [options.padding[0], 0],
            [options.padding_out[0], 0],
            [options.dilation[0], 1],
            options.groups,
        ),
        Default::default(),
    )?;
    let [batch_size, channels_out, height_out, _weight_out] = tensor.shape().dims();
    Ok(reshape(
        tensor,
        Shape::from([batch_size, channels_out, height_out]),
    ))
}
