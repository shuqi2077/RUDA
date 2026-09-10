use {ruda_core::tensor::spatial::ConvOptions};
use {ruda_core::tensor::Shape};
use {crate::convolution::AcceleratedTileKind, crate::convolution::ConvAlgorithm, crate::convolution::ConvolutionArgs, crate::convolution::ConvolutionInputs, crate::convolution::Strategy, crate::convolution::components::ConvSetupError, crate::convolution::launch_ref};
use {rublas::kernel_ir::definition::MatmulElems, rublas::kernel_ir::definition::MatmulGlobalElems};
use {ruda_kernel::tiling::InputBinding};

use {ruda_kernel::dsl::Runtime, ruda_kernel::tensor::allocation::empty_device_dtype, ruda_kernel::tensor::RudaTensor};

pub fn dgrad_gemm_simple_sync<R: Runtime, const N: usize>(
    out_grad: RudaTensor<R>,
    weights: RudaTensor<R>,
    input_shape: Shape,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<RudaTensor<R>, ConvSetupError> {
    let strategy = match tile_kind {
        AcceleratedTileKind::Cmma => Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleSyncCyclic,
            tile_kind,
        },
        AcceleratedTileKind::Mma => Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleSyncStrided,
            tile_kind,
        },
    };

    launch_backwards_data::<R, N>(&strategy, out_grad, weights, input_shape, options)
}

pub fn dgrad_gemm_simple_async<R: Runtime, const N: usize>(
    out_grad: RudaTensor<R>,
    weights: RudaTensor<R>,
    input_shape: Shape,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<RudaTensor<R>, ConvSetupError> {
    let strategy = match tile_kind {
        AcceleratedTileKind::Cmma => Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleAsyncCyclic,
            tile_kind,
        },
        AcceleratedTileKind::Mma => Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleAsyncStrided,
            tile_kind,
        },
    };

    launch_backwards_data::<R, N>(&strategy, out_grad, weights, input_shape, options)
}

pub fn dgrad_gemm_simple_tma<R: Runtime, const N: usize>(
    out_grad: RudaTensor<R>,
    weights: RudaTensor<R>,
    input_shape: Shape,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<RudaTensor<R>, ConvSetupError> {
    launch_backwards_data::<R, N>(
        &Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleAsyncTma,
            tile_kind,
        },
        out_grad,
        weights,
        input_shape,
        options,
    )
}

/// Perform a convolution backwards data pass using the implicit GEMM (im2col) algorithm, using
/// ruda tiling matmul components.
///
/// * `input` - The input feature map
/// * `out_grad` - The output gradients
/// * `weight_shape` - The shape of the weights/weight gradients
/// * `options` - The options to use for the convolution
pub fn launch_backwards_data<R: Runtime, const N: usize>(
    strategy: &Strategy,
    out_grad: RudaTensor<R>,
    weights: RudaTensor<R>,
    input_shape: Shape,
    options: ConvOptions<N>,
) -> Result<RudaTensor<R>, ConvSetupError> {
    if options.groups != 1 || options.stride.iter().any(|&s| s != 1) {
        return Err(ConvSetupError::Groups(options.groups));
    }

    let out_dtype = out_grad.dtype;

    let in_grad = empty_device_dtype(
        out_grad.client.clone(),
        out_grad.device.clone(),
        input_shape,
        out_dtype,
    );

    let client = out_grad.client.clone();
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        f32_math: Default::default(),
        lhs: out_grad.dtype.into(),
        rhs: weights.dtype.into(),
        out: out_dtype.into(),
    });
    let out_grad_dtype = out_grad.dtype;
    let weights_dtype = weights.dtype;
    let out_grad = InputBinding::new(out_grad.binding(), out_grad_dtype.into());
    let weights = InputBinding::new(weights.binding(), weights_dtype.into());

    launch_ref::<R, N>(
        strategy,
        &client,
        ConvolutionInputs::BackwardData {
            out_grad,
            weights,
            in_grad: in_grad.clone().binding(),
        },
        ConvolutionArgs {
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
        },
        dtypes,
    )?;

    Ok(in_grad)
}
