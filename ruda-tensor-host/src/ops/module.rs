//! Module operations for the Host backend.
//!
//! These operations power neural network modules like convolutions and pooling.

use crate::Host;
use ruda_tensor::{
    ops::{
        AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConv2dBackward,
        DeformConvOptions, InterpolateOptions,
        MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
    },
    tensor::{BoolTensor, FloatTensor, IntTensor},
};

pub(crate) use ruda_core::tensor::host::cast::{cast_from_f32, cast_to_f32};

impl ModuleOps<Host> for Host {
    fn conv1d(
        x: FloatTensor<Host>,
        weight: FloatTensor<Host>,
        bias: Option<FloatTensor<Host>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Host> {
        rudnn_host::convolution::dispatch::conv1d(x, weight, bias, options)
    }

    fn conv2d(
        x: FloatTensor<Host>,
        weight: FloatTensor<Host>,
        bias: Option<FloatTensor<Host>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Host> {
        rudnn_host::convolution::dispatch::conv2d(x, weight, bias, options)
    }

    fn deform_conv2d(
        x: FloatTensor<Host>,
        offset: FloatTensor<Host>,
        weight: FloatTensor<Host>,
        mask: Option<FloatTensor<Host>>,
        bias: Option<FloatTensor<Host>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Host> {
        rudnn_host::convolution::dispatch::deform_conv2d(x, offset, weight, mask, bias, options)
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Host>,
        offset: FloatTensor<Host>,
        weight: FloatTensor<Host>,
        mask: Option<FloatTensor<Host>>,
        bias: Option<FloatTensor<Host>>,
        output_grad: FloatTensor<Host>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Host> {
        let (x_grad, offset_grad, weight_grad, mask_grad, bias_grad) = rudnn_host::convolution::dispatch::deform_conv2d_backward(x, offset, weight, mask, bias, output_grad, options);
        DeformConv2dBackward::new(x_grad, offset_grad, weight_grad, mask_grad, bias_grad)
    }

    fn conv3d(
        x: FloatTensor<Host>,
        weight: FloatTensor<Host>,
        bias: Option<FloatTensor<Host>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Host> {
        rudnn_host::convolution::dispatch::conv3d(x, weight, bias, options)
    }

    fn conv_transpose1d(
        x: FloatTensor<Host>,
        weight: FloatTensor<Host>,
        bias: Option<FloatTensor<Host>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<Host> {
        rudnn_host::convolution::dispatch::conv_transpose1d(x, weight, bias, options)
    }

    fn conv_transpose2d(
        x: FloatTensor<Host>,
        weight: FloatTensor<Host>,
        bias: Option<FloatTensor<Host>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Host> {
        rudnn_host::convolution::dispatch::conv_transpose2d(x, weight, bias, options)
    }

    fn conv_transpose3d(
        x: FloatTensor<Host>,
        weight: FloatTensor<Host>,
        bias: Option<FloatTensor<Host>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Host> {
        rudnn_host::convolution::dispatch::conv_transpose3d(x, weight, bias, options)
    }

    fn avg_pool2d(
        x: FloatTensor<Host>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Host> {
        rudnn_host::pool::dispatch::avg_pool2d(x, kernel_size, stride, padding, count_include_pad, ceil_mode)
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Host>,
        grad: FloatTensor<Host>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        _divisor_override: bool,
    ) -> FloatTensor<Host> {
        rudnn_host::pool::dispatch::avg_pool2d_backward(x, grad, kernel_size, stride, padding, count_include_pad, _divisor_override)
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Host>, output_size: [usize; 2]) -> FloatTensor<Host> {
        rudnn_host::pool::dispatch::adaptive_avg_pool2d(x, output_size)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Host>,
        grad: FloatTensor<Host>,
    ) -> FloatTensor<Host> {
        rudnn_host::pool::dispatch::adaptive_avg_pool2d_backward(x, grad)
    }

    fn max_pool2d(
        x: FloatTensor<Host>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Host> {
        rudnn_host::pool::dispatch::max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Host>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Host> {
        let (output, indices) = rudnn_host::pool::dispatch::max_pool2d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode);
        MaxPool2dWithIndices::new(output, indices)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Host>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
        output_grad: FloatTensor<Host>,
        indices: IntTensor<Host>,
    ) -> MaxPool2dBackward<Host> {
        let x_grad = rudnn_host::pool::dispatch::max_pool2d_with_indices_backward(x, _kernel_size, _stride, _padding, _dilation, _ceil_mode, output_grad, indices);
        MaxPool2dBackward::new(x_grad)
    }

    fn interpolate(
        x: FloatTensor<Host>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Host> {
        rudnn_host::interpolate::dispatch::interpolate(x, output_size, options)
    }

    fn interpolate_backward(
        x: FloatTensor<Host>,
        grad: FloatTensor<Host>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Host> {
        rudnn_host::interpolate::dispatch::interpolate_backward(x, grad, output_size, options)
    }

    fn attention(
        query: FloatTensor<Host>,
        key: FloatTensor<Host>,
        value: FloatTensor<Host>,
        mask: Option<BoolTensor<Host>>,
        attn_bias: Option<FloatTensor<Host>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<Host> {
        crate::ops::attention::attention(query, key, value, mask, attn_bias, options)
    }

    fn rfft(
        signal: FloatTensor<Host>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Host>, FloatTensor<Host>) {
        rufft_host::dispatch::rfft(signal, dim, n)
    }

    fn irfft(
        spectrum_re: FloatTensor<Host>,
        spectrum_im: FloatTensor<Host>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Host> {
        rufft_host::dispatch::irfft(spectrum_re, spectrum_im, dim, n)
    }

    fn embedding(weights: FloatTensor<Host>, indices: IntTensor<Host>) -> FloatTensor<Host> {
        rudnn_host::embedding::embedding(weights, indices)
    }

    fn layer_norm(
        tensor: FloatTensor<Host>,
        gamma: FloatTensor<Host>,
        beta: Option<FloatTensor<Host>>,
        epsilon: f64,
    ) -> FloatTensor<Host> {
        crate::ops::activation::layer_norm(tensor, gamma, beta, epsilon)
    }

    fn embedding_backward(
        weights: FloatTensor<Host>,
        output_grad: FloatTensor<Host>,
        indices: IntTensor<Host>,
    ) -> FloatTensor<Host> {
        rudnn_host::embedding::embedding_backward(weights, output_grad, indices)
    }
}
