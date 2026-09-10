use {ruda_core::tensor::spatial::ConvTransposeOptions};
use {ruda_kernel::dsl::tune::LocalTuner, ruda_kernel::dsl::tune::Tunable, ruda_kernel::dsl::tune::TunableSet};

use {crate::convolution::tensor::tune_key::ConvTransposeTuneKey, ruda_kernel::dsl::Runtime, ruda_kernel::dsl::RudaTuneId, crate::convolution::tensor::ConvTranspose2dAutotuneKey, crate::convolution::tensor::conv_transpose2d_col2im, crate::convolution::tensor::conv_transpose2d_direct, ruda_kernel::tensor::RudaTensor};

/// Executes autotune on conv2d operations
pub fn conv_transpose2d_autotune<R: Runtime>(
    input: RudaTensor<R>,
    weights: RudaTensor<R>,
    bias: Option<RudaTensor<R>>,
    options: ConvTransposeOptions<2>,
) -> RudaTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<ConvTransposeTuneKey, RudaTuneId> = LocalTuner::new("ruda_tensor_device::kernel::conv::conv_transpose2d::tune::strict_f32_v1");

    let tune_set = TUNER.init(|| {
        TunableSet::new(create_key::<R>, create_transpose2d_input::<R>)
            .with(Tunable::new(
                "conv_transpose2d_direct",
                |(input, weights, bias, options)| {
                    conv_transpose2d_direct::<R>(input, weights, bias, options)
                },
            ))
            .with(Tunable::new(
                "conv_transpose2d_col2im",
                |(input, weights, bias, options)| {
                    conv_transpose2d_col2im::<R>(input, weights, bias, options)
                },
            ))
    });

    TUNER.execute(
        &RudaTuneId::new(&input.client, &input.device),
        &client,
        tune_set,
        (input, weights, bias, options),
    )
}

pub fn create_transpose2d_input<R: Runtime>(
    _key: &ConvTransposeTuneKey,
    (input, weights, bias, options): &(
        RudaTensor<R>,
        RudaTensor<R>,
        Option<RudaTensor<R>>,
        ConvTransposeOptions<2>,
    ),
) -> (
    RudaTensor<R>,
    RudaTensor<R>,
    Option<RudaTensor<R>>,
    ConvTransposeOptions<2>,
) {
    (
        input.clone(),
        weights.clone(),
        bias.clone(),
        options.clone(),
    )
}

fn create_key<R: Runtime>(
    (input, weights, bias, options): &(
        RudaTensor<R>,
        RudaTensor<R>,
        Option<RudaTensor<R>>,
        ConvTransposeOptions<2>,
    ),
) -> ConvTransposeTuneKey {
    let [batch_size, in_channels, height, width] = input.meta.shape().dims();
    let [out_channels, _, kernel_h, kernel_w] = weights.meta.shape().dims();
    let ConvTransposeOptions {
        stride,
        padding,
        dilation,
        groups,
        padding_out,
    } = options.clone();
    ConvTransposeTuneKey::ConvTranspose(ConvTranspose2dAutotuneKey::new(
        [kernel_h, kernel_w],
        stride,
        padding,
        padding_out,
        dilation,
        groups,
        in_channels,
        out_channels,
        height,
        width,
        batch_size,
        bias.is_some(),
        input.dtype,
    ))
}
