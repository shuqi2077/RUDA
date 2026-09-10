use {ruda_core::tensor::DType, ruda_core::tensor::spatial::ConvOptions, ruda_core::tensor::spatial::calculate_conv_output_sizes};
use {ruda_core::tensor::Metadata, ruda_core::tensor::Shape};
use {core::iter};
use {ruda_kernel::dsl::prelude::*, ruda_kernel::library::tensor::TensorHandle, ruda_kernel::library::tensor::into_contiguous_pitched};
use {crate::convolution::components::ConvSetupError};

use {ruda_kernel::dsl::Runtime, ruprim::elementwise::binary::numeric::AddOp, ruda_kernel::tensor::contiguous::into_contiguous_aligned, ruprim::elementwise::binary::numeric::launch_binop, rublas::tensor_matmul::MatmulStrategy, rublas::tensor_matmul::matmul, ruda_kernel::tensor::layout::split_dim, ruda_kernel::tensor::reshape::reshape, ruda_kernel::tensor::permutation::swap_dims, ruda_kernel::tensor::RudaTensor};

#[cfg(not(test))]
pub(crate) fn batches_per_run(
    batch_size: usize,
    out_shape: usize,
    plane_size: usize,
) -> Result<usize, ConvSetupError> {
    use {rublas::kernel_ir::definition::MatmulAvailabilityError};

    let ruda_count_per_batch = out_shape.div_ceil(plane_size);
    let max_ruda_count = u16::MAX as usize;
    let max_simultaneous = Ord::min(max_ruda_count / ruda_count_per_batch, batch_size);
    if max_simultaneous == 0 {
        return Err(MatmulAvailabilityError::RudaCountTooBig(RudaCount::Static(
            ruda_count_per_batch as u32,
            1,
            1,
        ))
        .into());
    }
    Ok((0..=max_simultaneous)
        .rev()
        .find(|per_run| batch_size.is_multiple_of(*per_run))
        .expect("Logically not possible"))
}

#[cfg(test)]
#[allow(unused)]
pub(crate) fn batches_per_run(
    batch_size: usize,
    out_shape: usize,
    plane_size: usize,
) -> Result<usize, ConvSetupError> {
    Ok(1)
}

pub fn conv_im2col_1x1<R: Runtime, const N: usize>(
    input: RudaTensor<R>,
    mut weight: RudaTensor<R>,
    bias: Option<RudaTensor<R>>,
    options: ConvOptions<N>,
) -> Result<RudaTensor<R>, ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

    let rank = input.meta.num_dims();
    let dim_c = rank - 1;

    let batch_size = input.meta.shape()[0];
    let in_channels = input.meta.shape()[dim_c];
    let in_shape = &input.meta.shape()[1..dim_c];
    let out_channels = weight.meta.shape()[0];
    let kernel_shape = &weight.meta.shape()[1..dim_c];

    if kernel_shape.iter().any(|s| *s != 1) {
        return Err(ConvSetupError::Unknown);
    }

    let out_shape = calculate_conv_output_sizes(
        kernel_shape,
        &options.stride,
        &options.padding,
        &options.dilation,
        in_shape,
    );

    let mut split_m = vec![batch_size];
    split_m.extend(out_shape.iter().copied());

    if kernel_shape.iter().any(|it| *it != 1) || in_shape != out_shape {
        return Err(ConvSetupError::Unknown);
    }

    let input = reshape_input(input); // [(NHW), C] : [M, K]
    let dtype = input.dtype;

    // Efficient permutation that takes the stride required for TMA into account
    let weight = if weight.meta.strides()[dim_c] != 1 {
        // Remove kernel dims so padded dim is channels
        *weight.meta = Metadata::new(
            [out_channels, in_channels], // [N, K]
            [weight.meta.strides()[0], weight.meta.strides()[dim_c]],
        );
        // Pitched contiguous to skip running another kernel for TMA
        into_contiguous_aligned(weight)
    } else {
        // Already compatible, skip initial reshape
        *weight.meta = Metadata::new([out_channels, in_channels], [weight.meta.strides()[0], 1]);
        weight
    };

    // Permute to N-major, while keeping memory layout K-major. K-major for both sides is the most
    // efficient for matmul, and allows skipping a contiguous kernel
    let weight = swap_dims(weight, 0, 1); // [K, N]

    let out = matmul(input, weight, None, MatmulStrategy::default(), dtype)?; // [M, N]

    // Skip reshape to avoid potential `into_contiguous`. We're only splitting dims so it's safe.
    let mut out = split_dim(out, 0, &split_m); // [N, H, W, C]

    if let Some(bias) = bias {
        let mut bias_shape = iter::repeat_n(1, rank - 1).collect::<Vec<_>>();
        bias_shape.push(out_channels);
        let bias = reshape(bias, bias_shape.into());
        out = launch_binop::<R, AddOp>(out, bias);
    }

    Ok(out)
}

/// Reshapes NHWC input to [(N, H, W), C]
fn reshape_input<R: Runtime>(input: RudaTensor<R>) -> RudaTensor<R> {
    let rank = input.meta.num_dims();
    let dim_c = rank - 1;
    let dtype = input.dtype;

    let batch_size = input.meta.shape()[0];
    let in_c: usize = input.meta.shape()[dim_c];
    let in_shape: Shape = input.meta.shape()[1..dim_c].into();

    let mut input = if !is_spatial_contiguous(input.meta.shape(), input.meta.strides()) {
        let (client, device) = (input.client.clone(), input.device.clone());
        let contiguous = into_contiguous_pitched(&client, input.binding(), dtype.into());
        from_handle(client, device, contiguous, dtype)
    } else {
        input
    };

    *input.meta = Metadata::new(
        [batch_size * in_shape.num_elements(), in_c], // [M, K]
        [input.meta.strides()[dim_c - 1], input.meta.strides()[dim_c]],
    );
    input
}

fn is_spatial_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    let rank = shape.len();
    let dim_c = rank - 1;

    // Channel must be contiguous for the [(N, H, W), C] reshape to be valid
    if strides[dim_c] != 1 {
        return false;
    }

    for i in (1..dim_c).rev() {
        if strides[i + 1] * shape[i + 1] != strides[i] {
            return false;
        }
    }
    true
}

fn from_handle<R: Runtime>(
    client: ComputeClient<R>,
    device: R::Device,
    handle: TensorHandle<R>,
    dtype: DType,
) -> RudaTensor<R> {
    RudaTensor::new(
        client.clone(),
        handle.handle,
        *handle.metadata,
        device.clone(),
        dtype,
    )
}
