use ruda_core::tensor::{DType, host::{HostTensor, cast::{cast_to_f32, cast_from_f32}}, spatial::{ConvOptions, ConvTransposeOptions, DeformConvOptions}};
use super::{forward as conv, transpose as conv_transpose, deformable as deform_conv};

pub fn conv1d(
    x: HostTensor,
    weight: HostTensor,
    bias: Option<HostTensor>,
    options: ConvOptions<1>,
) -> HostTensor {
    match x.dtype() {
        DType::F32 => conv::conv1d_f32(x, weight, bias, &options),
        DType::F64 => conv::conv1d_f64(x, weight, bias, &options),
        DType::F16 => conv::conv1d_f16(x, weight, bias, &options),
        DType::BF16 => conv::conv1d_bf16(x, weight, bias, &options),
        dtype => panic!("conv1d: unsupported dtype {:?}", dtype),
    }
}

pub fn conv2d(
    x: HostTensor,
    weight: HostTensor,
    bias: Option<HostTensor>,
    options: ConvOptions<2>,
) -> HostTensor {
    match x.dtype() {
        DType::F32 => conv::conv2d_f32(x, weight, bias, &options),
        DType::F64 => conv::conv2d_f64(x, weight, bias, &options),
        DType::F16 => conv::conv2d_f16(x, weight, bias, &options),
        DType::BF16 => conv::conv2d_bf16(x, weight, bias, &options),
        dtype => panic!("conv2d: unsupported dtype {:?}", dtype),
    }
}

pub fn deform_conv2d(
    x: HostTensor,
    offset: HostTensor,
    weight: HostTensor,
    mask: Option<HostTensor>,
    bias: Option<HostTensor>,
    options: DeformConvOptions<2>,
) -> HostTensor {
    let input_shape = x.layout().shape();
    let kernel_shape = weight.layout().shape();
    let output_size = options.output_size(
        [input_shape[2], input_shape[3]], [kernel_shape[2], kernel_shape[3]],
    );
    let offset_shape = offset.layout().shape();
    assert_eq!(
        [offset_shape[2], offset_shape[3]], output_size,
        "deform_conv2d offset spatial dimensions must match output"
    );
    match x.dtype() {
        DType::F32 => deform_conv::deform_conv2d_f32(
            x,
            offset,
            weight,
            mask,
            bias,
            options.stride,
            options.padding,
            options.dilation,
            options.weight_groups,
            options.offset_groups,
        ),
        DType::F64 => deform_conv::deform_conv2d_f64(
            x,
            offset,
            weight,
            mask,
            bias,
            options.stride,
            options.padding,
            options.dilation,
            options.weight_groups,
            options.offset_groups,
        ),
        DType::F16 => {
            use half::f16;
            let result = deform_conv::deform_conv2d_f32(
                cast_to_f32(x, f16::to_f32),
                cast_to_f32(offset, f16::to_f32),
                cast_to_f32(weight, f16::to_f32),
                mask.map(|m| cast_to_f32(m, f16::to_f32)),
                bias.map(|b| cast_to_f32(b, f16::to_f32)),
                options.stride,
                options.padding,
                options.dilation,
                options.weight_groups,
                options.offset_groups,
            );
            cast_from_f32(result, f16::from_f32)
        }
        DType::BF16 => {
            use half::bf16;
            let result = deform_conv::deform_conv2d_f32(
                cast_to_f32(x, bf16::to_f32),
                cast_to_f32(offset, bf16::to_f32),
                cast_to_f32(weight, bf16::to_f32),
                mask.map(|m| cast_to_f32(m, bf16::to_f32)),
                bias.map(|b| cast_to_f32(b, bf16::to_f32)),
                options.stride,
                options.padding,
                options.dilation,
                options.weight_groups,
                options.offset_groups,
            );
            cast_from_f32(result, bf16::from_f32)
        }
        dtype => panic!("deform_conv2d: unsupported dtype {:?}", dtype),
    }
}

pub fn deform_conv2d_backward(
    x: HostTensor,
    offset: HostTensor,
    weight: HostTensor,
    mask: Option<HostTensor>,
    bias: Option<HostTensor>,
    output_grad: HostTensor,
    options: DeformConvOptions<2>,
) -> (HostTensor, HostTensor, HostTensor, Option<HostTensor>, Option<HostTensor>) {
    let (x_grad, offset_grad, weight_grad, mask_grad, bias_grad) = match x.dtype() {
        DType::F32 => deform_conv::deform_conv2d_backward_f32(
            x,
            offset,
            weight,
            mask,
            bias,
            output_grad,
            options.stride,
            options.padding,
            options.dilation,
            options.weight_groups,
            options.offset_groups,
        ),
        DType::F16 => {
            use half::f16;
            let (xg, og, wg, mg, bg) = deform_conv::deform_conv2d_backward_f32(
                cast_to_f32(x, f16::to_f32),
                cast_to_f32(offset, f16::to_f32),
                cast_to_f32(weight, f16::to_f32),
                mask.map(|m| cast_to_f32(m, f16::to_f32)),
                bias.map(|b| cast_to_f32(b, f16::to_f32)),
                cast_to_f32(output_grad, f16::to_f32),
                options.stride,
                options.padding,
                options.dilation,
                options.weight_groups,
                options.offset_groups,
            );
            (
                cast_from_f32(xg, f16::from_f32),
                cast_from_f32(og, f16::from_f32),
                cast_from_f32(wg, f16::from_f32),
                mg.map(|m| cast_from_f32(m, f16::from_f32)),
                bg.map(|b| cast_from_f32(b, f16::from_f32)),
            )
        }
        DType::BF16 => {
            use half::bf16;
            let (xg, og, wg, mg, bg) = deform_conv::deform_conv2d_backward_f32(
                cast_to_f32(x, bf16::to_f32),
                cast_to_f32(offset, bf16::to_f32),
                cast_to_f32(weight, bf16::to_f32),
                mask.map(|m| cast_to_f32(m, bf16::to_f32)),
                bias.map(|b| cast_to_f32(b, bf16::to_f32)),
                cast_to_f32(output_grad, bf16::to_f32),
                options.stride,
                options.padding,
                options.dilation,
                options.weight_groups,
                options.offset_groups,
            );
            (
                cast_from_f32(xg, bf16::from_f32),
                cast_from_f32(og, bf16::from_f32),
                cast_from_f32(wg, bf16::from_f32),
                mg.map(|m| cast_from_f32(m, bf16::from_f32)),
                bg.map(|b| cast_from_f32(b, bf16::from_f32)),
            )
        }
        DType::F64 => deform_conv::deform_conv2d_backward_f64(
            x,
            offset,
            weight,
            mask,
            bias,
            output_grad,
            options.stride,
            options.padding,
            options.dilation,
            options.weight_groups,
            options.offset_groups,
        ),
        dtype => panic!("deform_conv2d_backward: unsupported dtype {:?}", dtype),
    };
    (x_grad, offset_grad, weight_grad, mask_grad, bias_grad)
}

pub fn conv3d(
    x: HostTensor,
    weight: HostTensor,
    bias: Option<HostTensor>,
    options: ConvOptions<3>,
) -> HostTensor {
    match x.dtype() {
        DType::F32 => conv::conv3d_f32(x, weight, bias, &options),
        DType::F64 => conv::conv3d_f64(x, weight, bias, &options),
        DType::F16 => conv::conv3d_f16(x, weight, bias, &options),
        DType::BF16 => conv::conv3d_bf16(x, weight, bias, &options),
        dtype => panic!("conv3d: unsupported dtype {:?}", dtype),
    }
}

pub fn conv_transpose1d(
    x: HostTensor,
    weight: HostTensor,
    bias: Option<HostTensor>,
    options: ConvTransposeOptions<1>,
) -> HostTensor {
    match x.dtype() {
        DType::F32 => conv_transpose::conv_transpose1d_f32(x, weight, bias, &options),
        DType::F64 => conv_transpose::conv_transpose1d_f64(x, weight, bias, &options),
        DType::F16 => conv_transpose::conv_transpose1d_f16(x, weight, bias, &options),
        DType::BF16 => conv_transpose::conv_transpose1d_bf16(x, weight, bias, &options),
        dtype => panic!("conv_transpose1d: unsupported dtype {:?}", dtype),
    }
}

pub fn conv_transpose2d(
    x: HostTensor,
    weight: HostTensor,
    bias: Option<HostTensor>,
    options: ConvTransposeOptions<2>,
) -> HostTensor {
    match x.dtype() {
        DType::F32 => conv_transpose::conv_transpose2d_f32(x, weight, bias, &options),
        DType::F64 => conv_transpose::conv_transpose2d_f64(x, weight, bias, &options),
        DType::F16 => conv_transpose::conv_transpose2d_f16(x, weight, bias, &options),
        DType::BF16 => conv_transpose::conv_transpose2d_bf16(x, weight, bias, &options),
        dtype => panic!("conv_transpose2d: unsupported dtype {:?}", dtype),
    }
}

pub fn conv_transpose3d(
    x: HostTensor,
    weight: HostTensor,
    bias: Option<HostTensor>,
    options: ConvTransposeOptions<3>,
) -> HostTensor {
    match x.dtype() {
        DType::F32 => conv_transpose::conv_transpose3d_f32(x, weight, bias, &options),
        DType::F64 => conv_transpose::conv_transpose3d_f64(x, weight, bias, &options),
        DType::F16 => conv_transpose::conv_transpose3d_f16(x, weight, bias, &options),
        DType::BF16 => conv_transpose::conv_transpose3d_bf16(x, weight, bias, &options),
        dtype => panic!("conv_transpose3d: unsupported dtype {:?}", dtype),
    }
}

