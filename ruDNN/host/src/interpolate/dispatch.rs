use ruda_core::tensor::{DType, host::HostTensor, spatial::{InterpolateMode, InterpolateOptions}};
use crate::interpolate;

pub fn interpolate(
    x: HostTensor,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> HostTensor {
    match (options.mode, x.dtype()) {
        (InterpolateMode::Nearest, DType::F32) => {
            interpolate::interpolate_nearest_f32(x, output_size, options.align_corners)
        }
        (InterpolateMode::Nearest, DType::F64) => {
            interpolate::interpolate_nearest_f64(x, output_size, options.align_corners)
        }
        (InterpolateMode::Nearest, DType::F16) => {
            interpolate::interpolate_nearest_f16(x, output_size, options.align_corners)
        }
        (InterpolateMode::Nearest, DType::BF16) => {
            interpolate::interpolate_nearest_bf16(x, output_size, options.align_corners)
        }
        (InterpolateMode::Bilinear, DType::F32) => {
            interpolate::interpolate_bilinear_f32(x, output_size, options.align_corners)
        }
        (InterpolateMode::Bilinear, DType::F64) => {
            interpolate::interpolate_bilinear_f64(x, output_size, options.align_corners)
        }
        (InterpolateMode::Bilinear, DType::F16) => {
            interpolate::interpolate_bilinear_f16(x, output_size, options.align_corners)
        }
        (InterpolateMode::Bilinear, DType::BF16) => {
            interpolate::interpolate_bilinear_bf16(x, output_size, options.align_corners)
        }
        (InterpolateMode::Bicubic, DType::F32) => {
            interpolate::interpolate_bicubic_f32(x, output_size, options.align_corners)
        }
        (InterpolateMode::Bicubic, DType::F64) => {
            interpolate::interpolate_bicubic_f64(x, output_size, options.align_corners)
        }
        (InterpolateMode::Bicubic, DType::F16) => {
            interpolate::interpolate_bicubic_f16(x, output_size, options.align_corners)
        }
        (InterpolateMode::Bicubic, DType::BF16) => {
            interpolate::interpolate_bicubic_bf16(x, output_size, options.align_corners)
        }
        (InterpolateMode::Lanczos3, DType::F32) => {
            interpolate::interpolate_lanczos3_f32(x, output_size, options.align_corners)
        }
        (InterpolateMode::Lanczos3, DType::F64) => {
            interpolate::interpolate_lanczos3_f64(x, output_size, options.align_corners)
        }
        (InterpolateMode::Lanczos3, DType::F16) => {
            interpolate::interpolate_lanczos3_f16(x, output_size, options.align_corners)
        }
        (InterpolateMode::Lanczos3, DType::BF16) => {
            interpolate::interpolate_lanczos3_bf16(x, output_size, options.align_corners)
        }
        (mode, dtype) => panic!(
            "interpolate: unsupported mode {:?} / dtype {:?}",
            mode, dtype
        ),
    }
}

pub fn interpolate_backward(
    x: HostTensor,
    grad: HostTensor,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> HostTensor {
    match (options.mode, x.dtype()) {
        (InterpolateMode::Nearest, DType::F32) => {
            interpolate::interpolate_nearest_backward_f32(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Nearest, DType::F64) => {
            interpolate::interpolate_nearest_backward_f64(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Nearest, DType::F16) => {
            interpolate::interpolate_nearest_backward_f16(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Nearest, DType::BF16) => {
            interpolate::interpolate_nearest_backward_bf16(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Bilinear, DType::F32) => {
            interpolate::interpolate_bilinear_backward_f32(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Bilinear, DType::F64) => {
            interpolate::interpolate_bilinear_backward_f64(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Bilinear, DType::F16) => {
            interpolate::interpolate_bilinear_backward_f16(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Bilinear, DType::BF16) => {
            interpolate::interpolate_bilinear_backward_bf16(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Bicubic, DType::F32) => {
            interpolate::interpolate_bicubic_backward_f32(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Bicubic, DType::F64) => {
            interpolate::interpolate_bicubic_backward_f64(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Bicubic, DType::F16) => {
            interpolate::interpolate_bicubic_backward_f16(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Bicubic, DType::BF16) => {
            interpolate::interpolate_bicubic_backward_bf16(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Lanczos3, DType::F32) => {
            interpolate::interpolate_lanczos3_backward_f32(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Lanczos3, DType::F64) => {
            interpolate::interpolate_lanczos3_backward_f64(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Lanczos3, DType::F16) => {
            interpolate::interpolate_lanczos3_backward_f16(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (InterpolateMode::Lanczos3, DType::BF16) => {
            interpolate::interpolate_lanczos3_backward_bf16(
                x,
                grad,
                output_size,
                options.align_corners,
            )
        }
        (mode, dtype) => {
            panic!(
                "interpolate_backward: unsupported mode {:?} / dtype {:?}",
                mode, dtype
            )
        }
    }
}

