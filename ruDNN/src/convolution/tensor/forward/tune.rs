use {ruda_core::tensor::spatial::ConvOptions};
use {ruda_kernel::dsl::ir::StorageType, ruda_kernel::dsl::tune::LocalTuner, ruda_kernel::dsl::tune::Tunable, ruda_kernel::dsl::tune::TunableSet, ruda_kernel::dsl::tune::anchor};
use {crate::convolution::AcceleratedTileKind};

use {crate::convolution::tensor::tune_key::ConvTuneKey, ruda_kernel::dsl::Runtime, ruda_kernel::dsl::RudaTuneId, crate::convolution::tensor::ConvAutotuneKey, crate::convolution::tensor::conv_direct, crate::convolution::tensor::conv_im2col_1x1, crate::convolution::tensor::forward::implicit_gemm::*, ruda_kernel::tensor::RudaTensor};

/// Executes autotune on convolution operations
pub fn conv_autotune<R: Runtime, const N: usize>(
    input: RudaTensor<R>,
    weight: RudaTensor<R>,
    bias: Option<RudaTensor<R>>,
    options: ConvOptions<N>,
) -> RudaTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<ConvTuneKey, RudaTuneId> = LocalTuner::new("ruda_tensor_device::kernel::conv::forward::tune::strict_f32_v1");

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key::<R, N>, create_conv_input::<R, N>)
            .with_stack_tuning(0, "convolution-whole-operator-v1", |(input, weight, bias, options)| {
                format!("input={};weight={};bias={:?};options={:?}",
                    input.autotune_signature(), weight.autotune_signature(),
                    bias.as_ref().map(|t| t.autotune_signature()), options)
            })
            .with(Tunable::new(
                "conv_direct",
                |(input, weight, bias, options)| conv_direct::<R, N>(input, weight, bias, options),
            ))
            .with(Tunable::new(
                "conv_im2col_1x1",
                |(input, weight, bias, options)| {
                    conv_im2col_1x1::<R, N>(input, weight, bias, options)
                },
            ))
            .with(Tunable::new(
                "simple_sync_cmma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_sync(input, weight, bias, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_sync_mma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_sync(input, weight, bias, options, AcceleratedTileKind::Mma)
                },
            ))
            .with(Tunable::new(
                "simple_async_cmma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_async(input, weight, bias, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_async_mma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_async(input, weight, bias, options, AcceleratedTileKind::Mma)
                },
            ))
            .with(Tunable::new(
                "simple_tma_cmma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_tma(input, weight, bias, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_tma_mma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_tma(input, weight, bias, options, AcceleratedTileKind::Mma)
                },
            ))
    });

    TUNER.execute(
        &RudaTuneId::new(&input.client, &input.device),
        &client,
        tunables,
        (input, weight, bias, options),
    )
}

pub fn create_conv_input<R: Runtime, const N: usize>(
    _key: &ConvTuneKey,
    (input, weights, bias, options): &(
        RudaTensor<R>,
        RudaTensor<R>,
        Option<RudaTensor<R>>,
        ConvOptions<N>,
    ),
) -> (
    RudaTensor<R>,
    RudaTensor<R>,
    Option<RudaTensor<R>>,
    ConvOptions<N>,
) {
    (
        input.clone(),
        weights.clone(),
        bias.clone(),
        options.clone(),
    )
}

fn create_key<R: Runtime, const N: usize>(
    (input, weights, bias, options): &(
        RudaTensor<R>,
        RudaTensor<R>,
        Option<RudaTensor<R>>,
        ConvOptions<N>,
    ),
) -> ConvTuneKey {
    let dtype = input.dtype;
    let rank = input.meta.shape().num_dims();
    let dim_c = rank - 1;

    let batch_size = input.meta.shape()[0];
    let in_channels = input.meta.shape()[dim_c];
    let out_channels = weights.meta.shape()[0];

    let kernel_size = weights.meta.shape()[1..dim_c].to_vec();
    let in_shape = input.meta.shape()[1..dim_c]
        .iter()
        .map(|shape| anchor(*shape, None, None, None))
        .collect();

    let ConvOptions {
        stride,
        padding,
        dilation,
        groups,
    } = options.clone();

    let lhs_stride_align = if input.meta.strides()[dim_c] == 1 {
        stride_align(input.meta.strides(), input.dtype.into())
    } else {
        0
    };
    let lhs_shape_align = pow2_factor(in_channels).min(lhs_stride_align);
    let rhs_stride_align = if weights.meta.strides()[dim_c] == 1 {
        stride_align(weights.meta.strides(), weights.dtype.into())
    } else {
        0
    };
    let rhs_shape_align = pow2_factor(in_channels).min(rhs_stride_align);

    ConvTuneKey::Conv(ConvAutotuneKey::new(
        kernel_size,
        stride.to_vec(),
        padding.to_vec(),
        dilation.to_vec(),
        groups,
        in_channels,
        out_channels,
        in_shape,
        batch_size,
        bias.is_some(),
        dtype,
        lhs_shape_align,
        lhs_stride_align,
        rhs_shape_align,
        rhs_stride_align,
    ))
}

/// Maximum factor relevant for strides. Currently set to 2^10 because that's 128-byte swizzle's
/// repeat number, so it's the largest align that can have performance impacts.
const MAX_STRIDE_FACTOR: u32 = 10;

/// Defines the non-contiguous stride alignment in terms of powers of two
fn stride_align(strides: &[usize], elem: StorageType) -> u8 {
    let max = MAX_STRIDE_FACTOR;
    let dim_c = strides.len() - 1;
    let factor = strides[..dim_c]
        .iter()
        .map(|it| (*it * elem.size_bits()) / 8)
        .map(|it| it.trailing_zeros())
        .min()
        .unwrap_or(max);
    factor.min(max) as u8
}

/// Defines the potential vectorization.
fn pow2_factor(axis: usize) -> u8 {
    axis.trailing_zeros().min(4) as u8
}
