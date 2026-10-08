use ruda_kernel::dsl::Runtime;
use ruda_kernel::dsl::prelude::StorageType;
use ruda_kernel::tensor::contiguous::into_contiguous;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::permutation::permute_nchw_to_nhwc;
use ruda_kernel::tensor::permutation::permute_nhwc_to_nchw;
use ruda_kernel::tensor::RudaTensor;
use ruda_core::tensor::Shape;
use ruda_core::tensor::DType;
use ruda_core::tensor::spatial::InterpolateMode;
use ruda_core::tensor::spatial::InterpolateOptions;

use super::{
    bicubic::interpolate_bicubic_launch, bilinear::interpolate_bilinear_launch,
    bicubic_backward::interpolate_bicubic_backward_launch,
    bilinear_backward::interpolate_bilinear_backward_launch,
    lanczos3::interpolate_lanczos3_launch, nearest::interpolate_nearest_launch,
    lanczos3_backward::interpolate_lanczos3_backward_launch,
    nearest_backward::interpolate_nearest_backward_launch,
};

pub(super) fn backward_types(output: DType, gradient: DType) -> [StorageType; 3] {
    let compute = if output == gradient { output }
        else if output == DType::F64 || gradient == DType::F64 { DType::F64 }
        else { DType::F32 };
    [output.into(), gradient.into(), compute.into()]
}

/// Interpolate operation
///
/// Supports nearest, bilinear, bicubic and lanczos3 modes
pub fn interpolate<R: Runtime>(
    input: RudaTensor<R>,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> RudaTensor<R> {
    let [batch_size, channels, _, _] = input.meta.shape().dims();
    let [out_height, out_width] = output_size;

    let input = into_contiguous(permute_nchw_to_nhwc(input));

    let shape_out = Shape::new([batch_size, out_height, out_width, channels]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape_out,
        input.dtype,
    );
    if output.meta.num_elements() == 0 { return permute_nhwc_to_nchw(output); }

    let align_corners = options.align_corners;
    let output = match options.mode {
        InterpolateMode::Nearest => interpolate_nearest_launch(input, output),
        InterpolateMode::Bilinear => interpolate_bilinear_launch(input, output, align_corners),
        InterpolateMode::Bicubic => interpolate_bicubic_launch(input, output, align_corners),
        InterpolateMode::Lanczos3 => interpolate_lanczos3_launch(input, output, align_corners),
    };

    permute_nhwc_to_nchw(output)
}

/// Backward interpolate operation
///
/// Supports nearest, bilinear, bicubic and lanczos3 modes.
pub fn interpolate_backward<R: Runtime>(
    input: RudaTensor<R>,
    out_grad: RudaTensor<R>,
    _output_size: [usize; 2],
    options: InterpolateOptions,
) -> RudaTensor<R> {
    backward_in_shape(&input, input.meta.shape().dims(), out_grad, options)
}

pub(super) fn backward_in_shape<R: Runtime>(reference: &RudaTensor<R>,
    input_shape: [usize; 4], out_grad: RudaTensor<R>, options: InterpolateOptions) -> RudaTensor<R> {
    let out_grad = permute_nchw_to_nhwc(out_grad);
    let [batch, channels, height, width] = input_shape;
    let output_shape = Shape::new([batch, height, width, channels]);
    let output = empty_device_dtype(
        reference.client.clone(),
        reference.device.clone(),
        output_shape,
        reference.dtype,
    );

    let output = match options.mode {
        InterpolateMode::Nearest => interpolate_nearest_backward_launch(out_grad, output),
        InterpolateMode::Bilinear => {
            interpolate_bilinear_backward_launch(out_grad, output, options.align_corners)
        }
        InterpolateMode::Bicubic => {
            interpolate_bicubic_backward_launch(out_grad, output, options.align_corners)
        }
        InterpolateMode::Lanczos3 => {
            interpolate_lanczos3_backward_launch(out_grad, output, options.align_corners)
        }
    };

    permute_nhwc_to_nchw(output)
}
