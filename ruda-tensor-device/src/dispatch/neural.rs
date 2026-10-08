use crate::{DeviceBackend, DeviceRuntime, FloatElement, IntElement, element::BoolElement};
use rudnn::convolution::tensor::ConvTranspose2dStrategy;
use ruda_tensor::tensor::{BoolTensor, FloatTensor, IntTensor};
use ruda_tensor::{
    TensorMetadata,
    ops::{
        AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConv2dBackward,
        DeformConvOptions, InterpolateOptions, MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
    },
};

fn norm_buffer<R: DeviceRuntime>(tensor: crate::RudaTensor<R>) -> ruda::runtime::normalization::TensorBuffer {
    ruda::runtime::normalization::TensorBuffer {
        shape: tensor.meta.shape().clone(), strides: tensor.meta.strides().clone(),
        handle: tensor.handle, dtype: tensor.dtype,
    }
}

fn norm_tensor<R: DeviceRuntime>(
    buffer: ruda::runtime::normalization::TensorBuffer,
    client: ruda::runtime::client::ComputeClient<R>, device: R::Device,
) -> crate::RudaTensor<R> {
    crate::RudaTensor::new(client, buffer.handle,
        ruda_core::tensor::Metadata::new(buffer.shape, buffer.strides), device, buffer.dtype)
}

fn native_norm_supported<R: DeviceRuntime>(tensor: &crate::RudaTensor<R>) -> bool {
    let properties = tensor.client.properties();
    let hardware = &properties.hardware;
    let plane = hardware.plane_size_max;
    matches!(tensor.dtype, ruda_core::tensor::DType::F32 | ruda_core::tensor::DType::F16 | ruda_core::tensor::DType::BF16)
        && tensor.meta.num_elements() <= u32::MAX as usize
        && tensor.meta.shape().last().is_some_and(|width| *width <= u32::MAX as usize)
        && plane.is_power_of_two() && plane == hardware.plane_size_min
        && properties.features.plane.contains(ruda_core::ir::features::Plane::Ops)
        && plane <= hardware.max_ruda_dim.0 && hardware.max_ruda_dim.1 >= 4
        && plane <= hardware.max_units_per_ruda / 4
        && tensor.meta.shape().last().is_some_and(|width| *width > 0
            && tensor.meta.num_elements() / width <= hardware.max_ruda_count.0 as usize)
}

fn native_rms_supported<R: DeviceRuntime>(tensor: &crate::RudaTensor<R>) -> bool {
    let properties = tensor.client.properties();
    let hardware = &properties.hardware;
    let plane = hardware.plane_size_max;
    matches!(tensor.dtype, ruda_core::tensor::DType::F32 | ruda_core::tensor::DType::F16 | ruda_core::tensor::DType::BF16)
        && tensor.meta.num_elements() <= u32::MAX as usize
        && properties.features.plane.contains(ruda_core::ir::features::Plane::Ops)
        && plane.is_power_of_two() && plane <= hardware.max_ruda_dim.0.min(hardware.max_units_per_ruda)
        && tensor.meta.shape().last().is_some_and(|width| *width > 0 && *width <= u32::MAX as usize
            && tensor.meta.num_elements() / width <= hardware.max_ruda_count.0 as usize)
}

impl<R, F, I, BT> ModuleOps<Self> for DeviceBackend<R, F, I, BT>
where
    R: DeviceRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn has_layer_norm_backward() -> bool { true }

    fn has_rms_norm_backward() -> bool { true }

    fn rms_norm_with_stats(tensor: FloatTensor<Self>, gamma: FloatTensor<Self>, epsilon: f64)
        -> ruda_tensor::ops::RmsNormOutput<Self> {
        if native_rms_supported(&tensor) && native_rms_supported(&gamma)
            && (epsilon as f32).is_finite() && (epsilon as f32) > 0.0 {
            let [output, rstd] = rudnn::normalization::rms_norm_with_stats(tensor, gamma, epsilon as f32)
                .expect("invalid native RMSNorm bindings");
            return ruda_tensor::ops::RmsNormOutput { output, rstd };
        }
        ruda_tensor::ops::normalization::rms_norm_with_stats::<Self>(tensor, gamma, epsilon)
    }

    fn rms_norm_backward(tensor: FloatTensor<Self>, gamma: FloatTensor<Self>, grad: FloatTensor<Self>,
        rstd: FloatTensor<Self>) -> ruda_tensor::ops::RmsNormBackward<Self> {
        if native_rms_supported(&tensor) && native_rms_supported(&gamma) && native_rms_supported(&grad)
            && rstd.dtype == ruda_core::tensor::DType::F32 {
            let [input, weight] = rudnn::normalization::rms_norm_backward(tensor, gamma, grad, rstd)
                .expect("invalid native RMSNorm backward bindings");
            return ruda_tensor::ops::RmsNormBackward { input, weight };
        }
        ruda_tensor::ops::normalization::rms_norm_backward::<Self>(tensor, gamma, grad, rstd)
    }

    fn layer_norm(
        tensor: FloatTensor<Self>, gamma: FloatTensor<Self>,
        beta: Option<FloatTensor<Self>>, epsilon: f64,
    ) -> FloatTensor<Self> {
        Self::layer_norm_with_stats(tensor, gamma, beta, epsilon).output
    }

    fn layer_norm_with_stats(
        tensor: FloatTensor<Self>, gamma: FloatTensor<Self>,
        beta: Option<FloatTensor<Self>>, epsilon: f64,
    ) -> ruda_tensor::ops::LayerNormOutput<Self> {
        if !R::has_native_layer_norm() {
            if native_norm_supported(&tensor) && native_norm_supported(&gamma)
                && beta.as_ref().is_none_or(native_norm_supported)
                && (epsilon as f32).is_finite() && (epsilon as f32) > 0.0 {
                let [output, mean, rstd] = rudnn::normalization::layer_norm_with_stats(
                    tensor, gamma, beta, epsilon as f32,
                ).expect("invalid native LayerNorm bindings");
                return ruda_tensor::ops::LayerNormOutput { output, mean, rstd };
            }
            return ruda_tensor::ops::normalization::layer_norm_with_stats::<Self>(tensor, gamma, beta, epsilon);
        }
        let client = tensor.client.clone();
        let device = tensor.device.clone();
        for other in core::iter::once(&gamma).chain(beta.iter()) {
            assert_eq!(device, other.device, "LayerNorm device mismatch");
            assert!(client.same_execution_queue(&other.client), "LayerNorm queue mismatch");
        }
        let [output, mean, rstd] = R::layer_norm(
            &client, norm_buffer(tensor), norm_buffer(gamma), beta.map(norm_buffer), epsilon,
        );
        let from = |buffer| norm_tensor(buffer, client.clone(), device.clone());
        ruda_tensor::ops::LayerNormOutput { output: from(output), mean: from(mean), rstd: from(rstd) }
    }

    fn layer_norm_backward(
        tensor: FloatTensor<Self>, gamma: FloatTensor<Self>, grad: FloatTensor<Self>,
        mean: FloatTensor<Self>, rstd: FloatTensor<Self>,
    ) -> ruda_tensor::ops::LayerNormBackward<Self> {
        if !R::has_native_layer_norm() {
            if native_norm_supported(&tensor) && native_norm_supported(&gamma) && native_norm_supported(&grad)
                && mean.dtype == ruda_core::tensor::DType::F32 && rstd.dtype == ruda_core::tensor::DType::F32 {
                let [input, weight, bias] = rudnn::normalization::layer_norm_backward(
                    tensor, gamma, grad, mean, rstd,
                ).expect("invalid native LayerNorm backward bindings");
                return ruda_tensor::ops::LayerNormBackward { input, weight, bias };
            }
            return ruda_tensor::ops::normalization::layer_norm_backward::<Self>(tensor, gamma, grad, mean, rstd);
        }
        let client = tensor.client.clone();
        let device = tensor.device.clone();
        for other in [&gamma, &grad, &mean, &rstd] {
            assert_eq!(device, other.device, "LayerNorm backward device mismatch");
            assert!(client.same_execution_queue(&other.client), "LayerNorm backward queue mismatch");
        }
        let [input, weight, bias] = R::layer_norm_backward(
            &client, norm_buffer(tensor), norm_buffer(gamma), norm_buffer(grad),
            norm_buffer(mean), norm_buffer(rstd),
        );
        let from = |buffer| norm_tensor(buffer, client.clone(), device.clone());
        ruda_tensor::ops::LayerNormBackward { input: from(input), weight: from(weight), bias: from(bias) }
    }

    fn conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        rudnn::convolution::tensor::conv_forward::<R, 1>(x, weight, bias, options, Default::default()).unwrap()
    }

    fn conv1d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        rudnn::convolution::tensor::conv_data_backward(
            output_grad,
            weight,
            x.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv1d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        rudnn::convolution::tensor::conv_weight_backward::<R, 1>(
            x,
            output_grad,
            weight.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        rudnn::convolution::tensor::conv_forward::<R, 2>(x, weight, bias, options, Default::default()).unwrap()
    }

    fn conv2d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        rudnn::convolution::tensor::conv_data_backward(
            output_grad,
            weight,
            x.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv2d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        rudnn::convolution::tensor::conv_weight_backward::<R, 2>(
            x,
            output_grad,
            weight.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        rudnn::convolution::tensor::deform_conv2d(x, offset, weight, mask, bias, options).unwrap()
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        let (x, o, w, m, b) = rudnn::convolution::tensor::deform_conv2d_backward(
            x,
            offset,
            weight,
            mask,
            bias,
            output_grad,
            options,
        )
        .unwrap();
        DeformConv2dBackward::new(x, o, w, m, b)
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        rudnn::convolution::tensor::conv_forward::<R, 3>(x, weight, bias, options, Default::default()).unwrap()
    }

    fn conv3d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        rudnn::convolution::tensor::conv_data_backward(
            output_grad,
            weight,
            x.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv3d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        rudnn::convolution::tensor::conv_weight_backward::<R, 3>(
            x,
            output_grad,
            weight.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        rudnn::convolution::tensor::conv_transpose2d(x, weight, bias, options, ConvTranspose2dStrategy::default())
            .unwrap()
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        rudnn::convolution::tensor::conv_transpose3d(x, weight, bias, options).expect("Kernel to never fail")
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        rudnn::pooling::avg_pool2d(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        rudnn::pooling::avg_pool2d_backward(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        rudnn::pooling::max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let (output, indices) = rudnn::pooling::max_pool2d_with_indices(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            I::dtype(),
        );

        MaxPool2dWithIndices::new(output, indices)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Self> {
        MaxPool2dBackward::new(rudnn::pooling::max_pool2d_with_indices_backward(
            x,
            output_grad,
            indices,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        ))
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        rudnn::pooling::adaptive_avg_pool2d(x, output_size)
    }

    fn adaptive_avg_pool3d(x: FloatTensor<Self>, output_size: [usize; 3]) -> FloatTensor<Self> {
        rudnn::pooling::adaptive_avg_pool3d(x, output_size)
    }

    fn max_pool3d(x: FloatTensor<Self>, kernel: [usize; 3], stride: [usize; 3],
        padding: [usize; 3], dilation: [usize; 3], ceil: bool) -> FloatTensor<Self> {
        rudnn::pooling::max_pool3d(x, kernel, stride, padding, dilation, ceil)
    }

    fn max_pool3d_with_indices(x: FloatTensor<Self>, kernel: [usize; 3], stride: [usize; 3],
        padding: [usize; 3], dilation: [usize; 3], ceil: bool) -> ruda_tensor::ops::MaxPool3dWithIndices<Self> {
        let (output, indices) = rudnn::pooling::max_pool3d_with_indices(x, kernel, stride, padding, dilation, ceil);
        ruda_tensor::ops::MaxPool3dWithIndices::new(output, indices)
    }

    fn max_pool3d_with_indices_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>, indices: IntTensor<Self>,
        kernel: [usize; 3], stride: [usize; 3], padding: [usize; 3], dilation: [usize; 3],
        ceil: bool) -> ruda_tensor::ops::MaxPool3dBackward<Self> {
        ruda_tensor::ops::MaxPool3dBackward::new(rudnn::pooling::max_pool3d_with_indices_backward(
            x, grad, indices, kernel, stride, padding, dilation, ceil))
    }

    fn avg_pool3d_native_output_size(input: [usize; 3], kernel: [usize; 3],
        stride: [usize; 3], padding: [usize; 3], ceil: bool) -> Option<[usize; 3]> {
        Some(rudnn::pooling::avg_pool3d_output_size(input, kernel, stride, padding, ceil))
    }

    fn avg_pool3d(x: FloatTensor<Self>, kernel: [usize; 3], stride: [usize; 3],
        padding: [usize; 3], include_pad: bool, ceil: bool) -> FloatTensor<Self> {
        rudnn::pooling::avg_pool3d(x, kernel, stride, padding, include_pad, ceil)
    }

    fn avg_pool3d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>, kernel: [usize; 3],
        stride: [usize; 3], padding: [usize; 3], include_pad: bool, ceil: bool) -> FloatTensor<Self> {
        rudnn::pooling::avg_pool3d_backward(x, grad, kernel, stride, padding, include_pad, ceil)
    }

    fn adaptive_avg_pool3d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        rudnn::pooling::adaptive_avg_pool3d_backward(x, grad)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        rudnn::pooling::adaptive_avg_pool2d_backward(x, grad)
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        rudnn::interpolation::interpolate(x, output_size, options)
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        rudnn::interpolation::interpolate_backward(x, grad, output_size, options)
    }

    fn interpolate1d(x: FloatTensor<Self>, size: usize, options: InterpolateOptions) -> FloatTensor<Self> {
        rudnn::interpolation::interpolate1d(x, size, options)
    }

    fn interpolate1d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>, size: usize,
        options: InterpolateOptions) -> FloatTensor<Self> {
        rudnn::interpolation::interpolate1d_backward(x, grad, size, options)
    }

    fn interpolate3d(x: FloatTensor<Self>, size: [usize; 3], options: InterpolateOptions) -> FloatTensor<Self> {
        rudnn::interpolation::interpolate3d(x, size, options)
    }

    fn interpolate3d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>, size: [usize; 3],
        options: InterpolateOptions) -> FloatTensor<Self> {
        rudnn::interpolation::interpolate3d_backward(x, grad, size, options)
    }

    fn attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
        attn_bias: Option<FloatTensor<Self>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<Self> {
        // Fall back to naive attention for features the flash kernel doesn't support.
        if attn_bias.is_some() || options.softcap.is_some() || options.scale.is_some() {
            return ruda_tensor::ops::attention::attention_fallback::<Self>(
                query, key, value, mask, attn_bias, options,
            );
        }

        rudnn::attention::tensor::attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            Default::default(),
        )
        .expect("Kernel to never fail")
    }

    fn has_ctc_loss_backward() -> bool {
        true
    }

    fn ctc_loss(
        log_probs: FloatTensor<Self>,
        targets: IntTensor<Self>,
        input_lengths: IntTensor<Self>,
        target_lengths: IntTensor<Self>,
        blank: usize,
    ) -> FloatTensor<Self> {
        rudnn::ctc::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank)
    }

    fn ctc_loss_backward(
        log_probs: FloatTensor<Self>,
        targets: IntTensor<Self>,
        input_lengths: IntTensor<Self>,
        target_lengths: IntTensor<Self>,
        grad_loss: FloatTensor<Self>,
        blank: usize,
    ) -> FloatTensor<Self> {
        let (log_alpha_full, log_beta_full, nll) = rudnn::ctc::ctc_alpha_beta(
            log_probs.clone(),
            targets.clone(),
            input_lengths.clone(),
            target_lengths,
            blank,
        );
        ruda_tensor::ops::ctc::ctc_grad_from_alpha_beta_default::<Self>(
            log_probs,
            targets,
            input_lengths,
            grad_loss,
            log_alpha_full,
            log_beta_full,
            nll,
            blank,
        )
    }

    fn rfft(
        signal: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        rufft::tensor::rfft(signal, dim, n)
    }

    fn irfft(
        spectrum_re: FloatTensor<Self>,
        spectrum_im: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Self> {
        rufft::tensor::irfft(spectrum_re, spectrum_im, dim, n)
    }
}
