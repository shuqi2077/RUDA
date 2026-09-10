use {ruda_core::tensor::spatial::ConvOptions};
use {ruda_core::tensor::Shape};
use {crate::convolution::AcceleratedTileKind, crate::convolution::components::ConvSetupError};

#[cfg(feature = "tensor-convolution-autotune")]
use {crate::convolution::tensor::backward_weight::wgrad_autotune, crate::convolution::tensor::dgrad_autotune};
use {ruda_kernel::dsl::Runtime, crate::convolution::tensor::backward_data::fallback::conv_data_backward_fallback, crate::convolution::tensor::backward_data::implicit_gemm::*, crate::convolution::tensor::backward_weight::fallback::conv_weight_backward_fallback, crate::convolution::tensor::backward_weight::implicit_gemm::*, crate::convolution::tensor::forward::implicit_gemm::conv_gemm_simple_sync, ruda_kernel::tensor::permutation::permute_nchw_to_nhwc, ruda_kernel::tensor::permutation::permute_nchw_to_nhwc_shape, ruda_kernel::tensor::permutation::permute_nhwc_to_nchw, ruda_kernel::tensor::RudaTensor};

use {super::conv_direct};
#[cfg(feature = "tensor-convolution-autotune")]
use {super::forward::conv_autotune};

/// The strategy to be used when launching a convolution kernel.
pub enum ConvStrategy {
    /// A simple direct convolution.
    Direct,
    #[cfg(feature = "tensor-convolution-autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
    /// Implicit GEMM implementation of convolution. Lower memory usage but requires CMMA and
    /// has constraints on tensor shape.
    ImplicitGemm,
}

impl Default for ConvStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "tensor-convolution-autotune")]
        return ConvStrategy::Autotune;

        // if autotune is disabled, default to the more memory-conservative algorithm
        #[cfg(not(feature = "tensor-convolution-autotune"))]
        ConvStrategy::Direct
    }
}

/// Performs an N-dimensional convolution with the given strategy
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
/// * `strategy` - The convolution algorithm to use. Autotune will pick the fastest available option.
pub fn conv_forward<R: Runtime, const N: usize>(
    input: RudaTensor<R>,
    weight: RudaTensor<R>,
    bias: Option<RudaTensor<R>>,
    options: ConvOptions<N>,
    strategy: ConvStrategy,
) -> Result<RudaTensor<R>, ConvSetupError> {
    let input = permute_nchw_to_nhwc(input);
    let weight = permute_nchw_to_nhwc(weight);

    let out = conv_forward_nhwc(input, weight, bias, options, strategy)?;

    Ok(permute_nhwc_to_nchw(out))
}

/// Performs an N-dimensional convolution with the given strategy on NHWC inputs/outputs
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
/// * `strategy` - The convolution algorithm to use. Autotune will pick the fastest available option.
pub fn conv_forward_nhwc<R: Runtime, const N: usize>(
    input: RudaTensor<R>,
    weight: RudaTensor<R>,
    bias: Option<RudaTensor<R>>,
    options: ConvOptions<N>,
    strategy: ConvStrategy,
) -> Result<RudaTensor<R>, ConvSetupError> {
    let strategy = if N == 3 && input.dtype == ruda_core::tensor::DType::F32 {
        ConvStrategy::Direct
    } else { strategy };
    match strategy {
        ConvStrategy::Direct => conv_direct::<R, N>(input, weight, bias, options),
        #[cfg(feature = "tensor-convolution-autotune")]
        ConvStrategy::Autotune => Ok(conv_autotune::<R, N>(input, weight, bias, options)),
        ConvStrategy::ImplicitGemm => {
            if options.groups != 1 {
                conv_direct::<R, N>(input, weight, bias, options)
            } else {
                conv_gemm_simple_sync::<R, N>(
                    input,
                    weight,
                    bias,
                    options,
                    AcceleratedTileKind::Cmma,
                )
            }
        }
    }
}

/// Performs an N-dimensional convolution backwards pass with regard to weight, with the given strategy
///
/// * `input` - The input feature map
/// * `out_grad` - The output gradients
/// * `weight_shape` - The shape of the weights/weight gradients
/// * `options` - The options used for the convolution
/// * `strategy` - The convolution algorithm to use. Autotune will pick the fastest available option.
pub fn conv_weight_backward<R: Runtime, const N: usize>(
    input: RudaTensor<R>,
    out_grad: RudaTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N>,
    strategy: ConvStrategy,
) -> Result<RudaTensor<R>, ConvSetupError> {
    let strategy = if N == 3 && input.dtype == ruda_core::tensor::DType::F32 {
        ConvStrategy::Direct
    } else { strategy };
    let input = permute_nchw_to_nhwc(input);
    let out_grad = permute_nchw_to_nhwc(out_grad);
    let weight_shape = permute_nchw_to_nhwc_shape(weight_shape);

    let weight_grad = match strategy {
        ConvStrategy::Direct => {
            conv_weight_backward_fallback::<R, N>(input, out_grad, weight_shape, options)
        }
        #[cfg(feature = "tensor-convolution-autotune")]
        ConvStrategy::Autotune => Ok(wgrad_autotune::<R, N>(
            input,
            out_grad,
            weight_shape,
            options,
        )),
        ConvStrategy::ImplicitGemm => {
            if options.groups != 1 {
                conv_weight_backward_fallback::<R, N>(input, out_grad, weight_shape, options)
            } else {
                wgrad_gemm_simple_sync::<R, N>(
                    input,
                    out_grad,
                    weight_shape,
                    options,
                    AcceleratedTileKind::Cmma,
                )
            }
        }
    }?;

    Ok(permute_nhwc_to_nchw(weight_grad))
}

/// Performs an N-dimensional convolution backwards data pass with the given strategy
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `in_shape` - The shape of the input to the layer
/// * `options` - The options to use for the convolution
/// * `strategy` - The convolution algorithm to use. Autotune will pick the fastest available option.
pub fn conv_data_backward<R: Runtime, const N: usize>(
    out_grad: RudaTensor<R>,
    weights: RudaTensor<R>,
    in_shape: Shape,
    options: ConvOptions<N>,
    strategy: ConvStrategy,
) -> Result<RudaTensor<R>, ConvSetupError> {
    let strategy = if N == 3 && out_grad.dtype == ruda_core::tensor::DType::F32 {
        ConvStrategy::Direct
    } else { strategy };
    let out_grad = permute_nchw_to_nhwc(out_grad);
    let weights = permute_nchw_to_nhwc(weights);
    let in_shape = permute_nchw_to_nhwc_shape(in_shape);

    let weight_grad = match strategy {
        ConvStrategy::Direct => {
            conv_data_backward_fallback::<R, N>(out_grad, weights, in_shape, options)?
        }
        #[cfg(feature = "tensor-convolution-autotune")]
        ConvStrategy::Autotune => dgrad_autotune::<R, N>(out_grad, weights, in_shape, options),
        ConvStrategy::ImplicitGemm => {
            if options.groups != 1 || options.stride.iter().any(|&s| s != 1) {
                conv_data_backward_fallback::<R, N>(out_grad, weights, in_shape, options)?
            } else {
                dgrad_gemm_simple_sync::<R, N>(
                    out_grad,
                    weights,
                    in_shape,
                    options,
                    AcceleratedTileKind::Cmma,
                )?
            }
        }
    };

    Ok(permute_nhwc_to_nchw(weight_grad))
}
