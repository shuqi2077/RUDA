use {ruda_core::tensor::spatial::ConvOptions};
use {ruda_core::tensor::Shape};
use {crate::convolution::AcceleratedTileKind, crate::convolution::ConvAlgorithm, crate::convolution::ConvolutionArgs, crate::convolution::ConvolutionInputs, crate::convolution::Strategy, crate::convolution::components::ConvSetupError, crate::convolution::launch_ref};
use {rublas::kernel_ir::definition::MatmulElems, rublas::kernel_ir::definition::MatmulGlobalElems};
use {ruda_kernel::tiling::InputBinding};

use {ruda_kernel::dsl::Runtime, ruda_kernel::tensor::allocation::empty_device_dtype, ruda_kernel::tensor::RudaTensor};

pub(crate) fn wgrad_gemm_simple_sync<R: Runtime, const N: usize>(
    input: RudaTensor<R>,
    out_grad: RudaTensor<R>,
    weight_shape: Shape,
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

    launch_backwards_weight::<R, N>(&strategy, input, out_grad, weight_shape, options)
}

pub(crate) fn wgrad_gemm_simple_async<R: Runtime, const N: usize>(
    input: RudaTensor<R>,
    out_grad: RudaTensor<R>,
    weight_shape: Shape,
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

    launch_backwards_weight::<R, N>(&strategy, input, out_grad, weight_shape, options)
}

pub(crate) fn wgrad_gemm_simple_tma<R: Runtime, const N: usize>(
    input: RudaTensor<R>,
    out_grad: RudaTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<RudaTensor<R>, ConvSetupError> {
    launch_backwards_weight::<R, N>(
        &Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleAsyncTma,
            tile_kind,
        },
        input,
        out_grad,
        weight_shape,
        options,
    )
}

/// Perform a convolution backwards weight pass using the implicit GEMM (im2col) algorithm, using
/// ruda tiling matmul components.
///
/// * `input` - The input feature map
/// * `out_grad` - The output gradients
/// * `weight_shape` - The shape of the weights/weight gradients
/// * `options` - The options to use for the convolution
pub fn launch_backwards_weight<R: Runtime, const N: usize>(
    strategy: &Strategy,
    input: RudaTensor<R>,
    out_grad: RudaTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N>,
) -> Result<RudaTensor<R>, ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

    let out_dtype = out_grad.dtype;

    let weight_grad = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        weight_shape,
        out_dtype,
    );

    let client = input.client.clone();
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        f32_math: Default::default(),
        lhs: input.dtype.into(),
        rhs: out_grad.dtype.into(),
        out: out_dtype.into(),
    });
    let input_dtype = input.dtype;
    let out_grad_dtype = out_grad.dtype;
    let input = InputBinding::new(input.binding(), input_dtype.into());
    let out_grad = InputBinding::new(out_grad.binding(), out_grad_dtype.into());

    launch_ref::<R, N>(
        strategy,
        &client,
        ConvolutionInputs::BackwardWeight {
            input,
            out_grad,
            weight_grad: weight_grad.clone().binding(),
        },
        ConvolutionArgs {
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
        },
        dtypes,
    )?;

    Ok(weight_grad)
}
