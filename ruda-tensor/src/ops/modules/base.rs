use super::{conv, ctc, embedding, interpolation, linear, pool};
use crate::ops::unfold::unfold4d_using_conv2d;
use crate::tensor::{BoolTensor, FloatTensor, IntTensor};
use crate::{Backend, ElementConversion, TensorMetadata};
use ruda_core::tensor::Shape;

/// Softmax/log-softmax output and its unrounded working-storage value.
pub struct SoftmaxOutput<B: Backend> {
    /// Output in the original input storage dtype.
    pub output: FloatTensor<B>,
    /// FP32 output, or FP64 for FP64 inputs, retained for first-order backward.
    pub working: FloatTensor<B>,
}

/// LayerNorm output and saved statistics used by its native backward operation.
pub struct LayerNormOutput<B: Backend> {
    /// Affine-normalized output.
    pub output: FloatTensor<B>,
    /// Per-row mean.
    pub mean: FloatTensor<B>,
    /// Per-row reciprocal standard deviation, including epsilon.
    pub rstd: FloatTensor<B>,
}

/// First-order LayerNorm gradients.
pub struct LayerNormBackward<B: Backend> {
    /// Input gradient.
    pub input: FloatTensor<B>,
    /// Scale gradient, summed over all leading dimensions.
    pub weight: FloatTensor<B>,
    /// Bias gradient, summed over all leading dimensions in the accumulation dtype.
    pub bias: FloatTensor<B>,
}

/// RMSNorm output and reciprocal row norms retained for first-order backward.
pub struct RmsNormOutput<B: Backend> {
    /// Affine-normalized output in the input storage dtype.
    pub output: FloatTensor<B>,
    /// Reciprocal row norms in the working dtype, including epsilon.
    pub rstd: FloatTensor<B>,
}

/// RMSNorm input and affine-weight gradients.
pub struct RmsNormBackward<B: Backend> {
    /// Input gradient in the original activation storage.
    pub input: FloatTensor<B>,
    /// Weight gradient in the original affine storage.
    pub weight: FloatTensor<B>,
}

/// Gradient computed during the backward pass for each tensor used by [conv2d](ModuleOps::conv2d).
#[derive(new)]
pub struct Conv2dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B>,

    /// Weights gradient.
    pub weights_grad: FloatTensor<B>,

    /// Bias gradient.
    pub bias_grad: Option<FloatTensor<B>>,
}

/// Gradient computed during the backward pass for each tensor used by [deform_conv2d](ModuleOps::deform_conv2d).
#[derive(new)]
pub struct DeformConv2dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B>,

    /// Offset gradient.
    pub offset_grad: FloatTensor<B>,

    /// Weights gradient.
    pub weight_grad: FloatTensor<B>,

    /// Mask gradient.
    pub mask_grad: Option<FloatTensor<B>>,

    /// Bias gradient.
    pub bias_grad: Option<FloatTensor<B>>,
}

/// Gradient computed during the backward pass for each tensor used by [conv3d](ModuleOps::conv3d).
#[derive(new)]
pub struct Conv3dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B>,

    /// Weights gradient.
    pub weights_grad: FloatTensor<B>,

    /// Bias gradient.
    pub bias_grad: Option<FloatTensor<B>>,
}

/// Gradient computed during the backward pass for each tensor used by [max_pool1d](ModuleOps::max_pool1d).
#[derive(new)]
pub struct MaxPool1dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B>,
}

/// Results from [max_pool1d](ModuleOps::max_pool1d_with_indices).
#[derive(new)]
pub struct MaxPool1dWithIndices<B: Backend> {
    /// The output tensor.
    pub output: FloatTensor<B>,

    /// The indices tensor.
    pub indices: IntTensor<B>,
}

/// Gradient computed during the backward pass for each tensor used by [max_pool2d](ModuleOps::max_pool2d).
#[derive(new)]
pub struct MaxPool2dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B>,
}

/// Results from [max_pool2d](ModuleOps::max_pool2d_with_indices).
#[derive(new)]
pub struct MaxPool2dWithIndices<B: Backend> {
    /// The output tensor.
    pub output: FloatTensor<B>,

    /// The indices tensor.
    pub indices: IntTensor<B>,
}

/// Native volume maxima and flattened I64 input positions.
#[derive(new)]
pub struct MaxPool3dWithIndices<B: Backend> {
    /// Output `[batch, channels, depth, height, width]` activations.
    pub output: FloatTensor<B>,
    /// Original input positions per batch/channel, or `-1` for empty selections.
    pub indices: IntTensor<B>,
}

/// Native volume maximum-pooling input gradients.
#[derive(new)]
pub struct MaxPool3dBackward<B: Backend> {
    /// Gradient in the original input geometry and storage.
    pub x_grad: FloatTensor<B>,
}

pub use ruda_core::tensor::spatial::{ConvOptions, PaddedConvOptions, DeformConvOptions, ConvTransposeOptions, UnfoldOptions};

pub use ruda_core::tensor::spatial::{InterpolateMode, InterpolateOptions};

pub use ruda_core::tensor::spatial::{GridSampleOptions, GridSamplePaddingMode};

/// Padding mode for tensor pad operations.
///
/// Defines how values are filled when padding a tensor beyond its original boundaries.
/// Padding can be applied to any dimension of a tensor.
///
/// # Modes
///
/// - [`Constant`](PadMode::Constant): Fill with a specified value (default: 0.0)
/// - [`Reflect`](PadMode::Reflect): Mirror values at boundary, excluding edge (requires padding < dim_size)
/// - [`Edge`](PadMode::Edge): Replicate boundary values
#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum PadMode {
    /// Fill padded regions with a constant value.
    ///
    /// # Example
    /// For tensor `[1, 2, 3]` with padding 2 on the left and value 0:
    /// Result: `[0, 0, 1, 2, 3]`
    Constant(f32),

    /// Reflect values at the boundary, excluding the edge value.
    ///
    /// Padding must be less than the dimension size (i.e., `padding < dim_size`).
    ///
    /// # Example
    /// For tensor `[1, 2, 3, 4]` with padding 2 on the left:
    /// Result: `[3, 2, 1, 2, 3, 4]` (reflects from index 1, not 0)
    Reflect,

    /// Replicate the edge values.
    ///
    /// # Example
    /// For tensor `[1, 2, 3, 4]` with padding 2 on the left:
    /// Result: `[1, 1, 1, 2, 3, 4]`
    Edge,
}

impl Default for PadMode {
    fn default() -> Self {
        PadMode::Constant(0.0)
    }
}

impl<E: ElementConversion> From<E> for PadMode {
    fn from(value: E) -> Self {
        PadMode::Constant(value.elem())
    }
}

/// Gradient computed during the backward pass for each tensor used by [interpolate](ModuleOps::interpolate).
#[derive(new)]
pub struct InterpolateBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B>,
}

pub use ruda_core::tensor::spatial::AttentionModuleOptions;

/// Module operations trait.
pub trait ModuleOps<B: Backend> {
    /// Explicit saved-working-output softmax path; existing activation defaults are unchanged.
    fn softmax_native(tensor: FloatTensor<B>, dim: usize, logarithmic: bool) -> FloatTensor<B> {
        Self::softmax_with_stats(tensor, dim, logarithmic).output
    }

    /// Softmax/log-softmax retaining its working output, with FP32 low-precision arithmetic.
    fn softmax_with_stats(tensor: FloatTensor<B>, dim: usize, logarithmic: bool) -> SoftmaxOutput<B> {
        super::softmax::softmax_with_stats::<B>(tensor, dim, logarithmic)
    }

    /// VJP from the saved working output, retaining FP64 if either argument uses FP64.
    fn softmax_native_backward(working: FloatTensor<B>, grad: FloatTensor<B>, dim: usize,
        logarithmic: bool) -> FloatTensor<B> {
        super::softmax::softmax_backward::<B>(working, grad, dim, logarithmic)
    }

    /// Embedding operation.
    ///
    /// # Arguments
    ///
    /// * `weights` - The embedding weights.
    /// * `indices` - The indices tensor.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn embedding(weights: FloatTensor<B>, indices: IntTensor<B>) -> FloatTensor<B> {
        embedding::embedding::<B>(weights, indices)
    }

    /// Embedding backward operation.
    ///
    /// # Arguments
    ///
    /// * `weights` - The embedding weights.
    /// * `output_grad` - The output gradient.
    /// * `indices` - The indices tensor.
    ///
    /// # Returns
    ///
    /// The gradient.
    fn embedding_backward(
        weights: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        indices: IntTensor<B>,
    ) -> FloatTensor<B> {
        embedding::embedding_backward::<B>(weights, output_grad, indices)
    }

    /// Linear transformation.
    ///
    /// # Shapes
    ///
    /// x:      `[..., d_input]`,
    /// weight: `[d_input, d_output]`,
    /// bias:   `[d_output]`,
    fn linear(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
    ) -> FloatTensor<B> {
        linear::linear::<B>(x, weight, bias)
    }
    /// Backward pass for [linear](ModuleOps::linear), returning the gradient for `x`.
    fn linear_x_backward(weight: FloatTensor<B>, output_grad: FloatTensor<B>) -> FloatTensor<B> {
        linear::linear_x_backward::<B>(weight, output_grad)
    }
    /// Backward pass for [linear](ModuleOps::linear), returning the gradient for `weight`.
    fn linear_weight_backward(x: FloatTensor<B>, output_grad: FloatTensor<B>) -> FloatTensor<B> {
        linear::linear_weight_backward::<B>(x, output_grad)
    }
    /// Backward pass for [linear](ModuleOps::linear), returning the gradient for `bias`.
    fn linear_bias_backward(output_grad: FloatTensor<B>) -> FloatTensor<B> {
        linear::linear_bias_backward::<B>(output_grad)
    }

    /// One dimensional convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, length]`,
    /// weight: `[channels_out, channels_in, kernel_size]`,
    /// bias:   `[channels_out]`,
    fn conv1d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<B> {
        conv::conv1d_from_conv2d::<B>(x, weight, bias, options)
    }
    /// Backward pass for the [conv1d](ModuleOps::conv1d) operation, returning the gradient for `x`.
    fn conv1d_x_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<1>,
    ) -> FloatTensor<B> {
        conv::conv1d_x_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv1d](ModuleOps::conv1d) operation, returning the gradient for `weight`.
    fn conv1d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<1>,
    ) -> FloatTensor<B> {
        conv::conv1d_weight_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv1d](ModuleOps::conv1d) operation, returning the gradient for `bias`.
    fn conv1d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        conv::conv1d_bias_backward::<B>(x, bias, output_grad)
    }
    /// Two dimensional convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, height, width]`,
    /// weight: `[channels_out, channels_in, kernel_size_1, kernel_size_2]`,
    /// bias:   `[channels_out]`,
    fn conv2d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<B>;
    /// Backward pass for the [conv2d](ModuleOps::conv2d) operation, returning the gradient for `x`.
    fn conv2d_x_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<2>,
    ) -> FloatTensor<B> {
        conv::conv2d_x_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv2d](ModuleOps::conv2d) operation, returning the gradient for `weight`.
    fn conv2d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<2>,
    ) -> FloatTensor<B> {
        conv::conv2d_weight_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv2d](ModuleOps::conv2d) operation, returning the gradient for `bias`.
    fn conv2d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        conv::conv2d_bias_backward::<B>(x, bias, output_grad)
    }

    /// Two dimensional deformable convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, height, width]`,
    /// weight: `[channels_out, channels_in, kernel_size_1, kernel_size_2]`,
    /// bias:   `[channels_out]`,
    fn deform_conv2d(
        x: FloatTensor<B>,
        offset: FloatTensor<B>,
        weight: FloatTensor<B>,
        mask: Option<FloatTensor<B>>,
        bias: Option<FloatTensor<B>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<B>;
    /// Backward pass for the [deform_conv2d](ModuleOps::deform_conv2d) operation.
    fn deform_conv2d_backward(
        x: FloatTensor<B>,
        offset: FloatTensor<B>,
        weight: FloatTensor<B>,
        mask: Option<FloatTensor<B>>,
        bias: Option<FloatTensor<B>>,
        output_grad: FloatTensor<B>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<B>;

    /// Three dimensional convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, depth, height, width]`,
    /// weight: `[channels_out, channels_in, kernel_size_1, kernel_size_2, kernel_size_3]`,
    /// bias:   `[channels_out]`,
    fn conv3d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<B>;
    /// Backward pass for the [conv3d](ModuleOps::conv3d) operation, returning the gradient for `x`.
    fn conv3d_x_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<3>,
    ) -> FloatTensor<B> {
        conv::conv3d_x_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv3d](ModuleOps::conv3d) operation, returning the gradient for `weight`.
    fn conv3d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<3>,
    ) -> FloatTensor<B> {
        conv::conv3d_weight_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv3d](ModuleOps::conv3d) operation, returning the gradient for `bias`.
    fn conv3d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        conv::conv3d_bias_backward::<B>(x, bias, output_grad)
    }
    /// One dimensional transposed convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, length]`,
    /// weight: `[channels_in, channels_out, length]`,
    /// bias:   `[channels_out]`,
    fn conv_transpose1d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<B> {
        conv::conv_transpose1d_from_conv_transpose2d::<B>(x, weight, bias, options)
    }
    /// Backward pass for the [conv transpose 1d](ModuleOps::conv_transpose1d) operation, returning the gradient for `x`.
    fn conv_transpose1d_x_backward(
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<B> {
        conv::conv_transpose1d_x_backward::<B>(weight, output_grad, options)
    }
    /// Backward pass for the [conv transpose 1d](ModuleOps::conv_transpose1d) operation, returning the gradient for `weight`.
    fn conv_transpose1d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<B> {
        conv::conv_transpose1d_weight_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv transpose 1d](ModuleOps::conv_transpose1d) operation, returning the gradient for `bias`.
    fn conv_transpose1d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        conv::conv_transpose1d_bias_backward::<B>(x, bias, output_grad)
    }

    /// Two dimensional transposed convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, height, width]`,
    /// weight: `[channels_in, channels_out, kernel_size_1, kernel_size_2]`,
    /// bias:   `[channels_out]`,
    fn conv_transpose2d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<B>;
    /// Backward pass for the [conv transpose 2d](ModuleOps::conv_transpose2d) operation, returning the gradient for `x`.
    fn conv_transpose2d_x_backward(
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<B> {
        conv::conv_transpose2d_x_backward::<B>(weight, output_grad, options)
    }
    /// Backward pass for the [conv transpose 2d](ModuleOps::conv_transpose2d) operation, returning the gradient for `weight`.
    fn conv_transpose2d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<B> {
        conv::conv_transpose2d_weight_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv transpose 2d](ModuleOps::conv_transpose2d) operation, returning the gradient for `bias`.
    fn conv_transpose2d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        conv::conv_transpose2d_bias_backward::<B>(x, bias, output_grad)
    }

    /// Three dimensional transposed convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, height, width]`,
    /// weight: `[channels_in, channels_out, kernel_size_1, kernel_size_2, kernel_size_3]`,
    /// bias:   `[channels_out]`,
    fn conv_transpose3d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<B>;
    /// Backward pass for the [conv transpose 3d](ModuleOps::conv_transpose3d) operation, returning the gradient for `x`.
    fn conv_transpose3d_x_backward(
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<B> {
        conv::conv_transpose3d_x_backward::<B>(weight, output_grad, options)
    }
    /// Backward pass for the [conv transpose 3d](ModuleOps::conv_transpose3d) operation, returning the gradient for `weight`.
    fn conv_transpose3d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<B> {
        conv::conv_transpose3d_weight_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv transpose 3d](ModuleOps::conv_transpose3d) operation, returning the gradient for `bias`.
    fn conv_transpose3d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        conv::conv_transpose3d_bias_backward::<B>(x, bias, output_grad)
    }

    /// Four-dimensional unfolding.
    ///
    /// # Shapes
    ///
    /// * x:      ``[batch_size, channels_in, height, width]``,
    /// * returns: ``[batch_size, channels_in * kernel_size_1 * kernel_size_2, number of blocks]``,
    fn unfold4d(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        options: UnfoldOptions,
    ) -> FloatTensor<B> {
        if options.padding == [0, 0] && options.dilation == [1, 1] {
            let blocks = B::float_unfold(x, 2, kernel_size[0], options.stride[0]);
            let blocks = B::float_unfold(blocks, 3, kernel_size[1], options.stride[1]);

            // batch, channels, h_blocks, w_blocks, h_kern, w_kern

            let blocks = B::float_permute(blocks, &[0, 1, 4, 5, 2, 3]);
            let shape = blocks.shape();

            // batch, channels, h_kern, w_kern, h_blocks, w_blocks

            B::float_reshape(
                blocks,
                [
                    shape[0],
                    shape[1] * shape[2] * shape[3],
                    shape[4] * shape[5],
                ]
                .into(),
            )
        } else {
            unfold4d_using_conv2d::<B>(x, kernel_size, options)
        }
    }

    /// One dimensional avg pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, length],
    fn avg_pool1d(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<B> {
        pool::avg_pool1d_from_2d::<B>(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }
    /// Backward pass for the [avg pooling 1d](ModuleOps::avg_pool1d) operation.
    fn avg_pool1d_backward(
        x: FloatTensor<B>,
        grad: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<B> {
        pool::avg_pool1d_backward_from_2d::<B>(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }
    /// Two dimensional avg pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, height, width],
    fn avg_pool2d(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<B>;
    /// Backward pass for the [avg pooling 2d](ModuleOps::avg_pool2d) operation.
    fn avg_pool2d_backward(
        x: FloatTensor<B>,
        grad: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<B>;
    /// Three dimensional average pooling of `[batch, channels, depth, height, width]`.
    fn avg_pool3d(x: FloatTensor<B>, kernel_size: [usize; 3], stride: [usize; 3],
        padding: [usize; 3], count_include_pad: bool, ceil_mode: bool) -> FloatTensor<B> {
        pool::avg_pool3d_from_2d::<B>(x, kernel_size, stride, padding, count_include_pad, ceil_mode)
    }
    /// Native volume output geometry, or `None` to retain the backend's composed operations.
    fn avg_pool3d_native_output_size(_input: [usize; 3], _kernel: [usize; 3],
        _stride: [usize; 3], _padding: [usize; 3], _ceil: bool) -> Option<[usize; 3]> {
        None
    }
    /// Input gradients for three dimensional average pooling.
    fn avg_pool3d_backward(x: FloatTensor<B>, grad: FloatTensor<B>, kernel_size: [usize; 3],
        stride: [usize; 3], padding: [usize; 3], count_include_pad: bool, ceil_mode: bool) -> FloatTensor<B> {
        pool::avg_pool3d_backward_from_2d::<B>(x, grad, kernel_size, stride, padding, count_include_pad, ceil_mode)
    }
    /// Two dimensional adaptive avg pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, height, width],
    fn adaptive_avg_pool2d(x: FloatTensor<B>, output_size: [usize; 2]) -> FloatTensor<B>;
    /// Backward pass for the [adaptive avg pooling 2d](ModuleOps::adaptive_avg_pool2d) operation.
    fn adaptive_avg_pool2d_backward(x: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B>;
    /// Three dimensional adaptive average pooling of `[batch, channels, depth, height, width]`.
    fn adaptive_avg_pool3d(x: FloatTensor<B>, output_size: [usize; 3]) -> FloatTensor<B> {
        pool::adaptive_avg_pool3d_from_2d::<B>(x, output_size)
    }
    /// Input gradients for native three dimensional adaptive average pooling.
    fn adaptive_avg_pool3d_backward(x: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B> {
        pool::adaptive_avg_pool3d_backward_from_2d::<B>(x, grad)
    }
    /// Three dimensional maximum pooling with native padding, stride and dilation.
    fn max_pool3d(x: FloatTensor<B>, kernel: [usize; 3], stride: [usize; 3], padding: [usize; 3],
        dilation: [usize; 3], ceil: bool) -> FloatTensor<B> {
        pool::max_pool3d_from_2d::<B>(x, kernel, stride, padding, dilation, ceil)
    }
    /// Volume maxima and original input positions.
    fn max_pool3d_with_indices(x: FloatTensor<B>, kernel: [usize; 3], stride: [usize; 3],
        padding: [usize; 3], dilation: [usize; 3], ceil: bool) -> MaxPool3dWithIndices<B> {
        pool::max_pool3d_with_indices_from_2d::<B>(x, kernel, stride, padding, dilation, ceil)
    }
    /// Accumulate volume gradients through the saved maximum positions.
    fn max_pool3d_with_indices_backward(x: FloatTensor<B>, grad: FloatTensor<B>, indices: IntTensor<B>,
        _kernel: [usize; 3], _stride: [usize; 3], _padding: [usize; 3], _dilation: [usize; 3],
        _ceil: bool) -> MaxPool3dBackward<B> {
        MaxPool3dBackward::new(pool::max_pool3d_backward_from_indices::<B>(x, grad, indices))
    }
    /// One dimensional adaptive avg pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, length],
    fn adaptive_avg_pool1d(x: FloatTensor<B>, output_size: usize) -> FloatTensor<B> {
        pool::adaptive_avg_pool1d_from_2d::<B>(x, output_size)
    }
    /// Backward pass for the [adaptive avg pooling 1d](ModuleOps::adaptive_avg_pool1d) operation.
    fn adaptive_avg_pool1d_backward(x: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B> {
        pool::adaptive_avg_pool1d_backward_from_2d::<B>(x, grad)
    }
    /// One dimensional max pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, length],
    fn max_pool1d(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> FloatTensor<B> {
        pool::max_pool1d_from_2d::<B>(x, kernel_size, stride, padding, dilation, ceil_mode)
    }

    /// One dimensional max pooling with indices.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, height, width],
    fn max_pool1d_with_indices(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> MaxPool1dWithIndices<B> {
        pool::max_pool1d_with_indices_from_2d::<B>(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )
    }
    /// Backward pass for the [max pooling 1d](ModuleOps::max_pool1d_with_indices) operation.
    #[allow(clippy::too_many_arguments)]
    fn max_pool1d_with_indices_backward(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
        output_grad: FloatTensor<B>,
        indices: IntTensor<B>,
    ) -> MaxPool1dBackward<B> {
        pool::max_pool1d_with_indices_backward_from_2d::<B>(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad,
            indices,
        )
    }

    /// Two dimensional max pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, height, width],
    fn max_pool2d(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<B>;

    /// Two dimensional max pooling with indices.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, height, width],
    fn max_pool2d_with_indices(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<B>;
    /// Backward pass for the [max pooling 2d](ModuleOps::max_pool2d_with_indices) operation.
    #[allow(clippy::too_many_arguments)]
    fn max_pool2d_with_indices_backward(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: FloatTensor<B>,
        indices: IntTensor<B>,
    ) -> MaxPool2dBackward<B>;

    /// Down/up samples the input.
    ///
    /// # Shapes
    ///
    /// x: `[batch_size, channels, height, width]`,
    fn interpolate(
        x: FloatTensor<B>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<B>;

    /// Backward pass for the [interpolate](ModuleOps::interpolate) operation.
    fn interpolate_backward(
        x: FloatTensor<B>,
        grad: FloatTensor<B>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<B>;

    /// Resize native `[batch, channels, width]` lines with the original filter.
    fn interpolate1d(x: FloatTensor<B>, size: usize, options: InterpolateOptions) -> FloatTensor<B> {
        interpolation::interpolate1d_from_2d::<B>(x, size, options)
    }

    /// Backward line resizing in the original input geometry and storage.
    fn interpolate1d_backward(x: FloatTensor<B>, grad: FloatTensor<B>, size: usize,
        options: InterpolateOptions) -> FloatTensor<B> {
        interpolation::interpolate1d_backward_from_2d::<B>(x, grad, size, options)
    }

    /// Resize native volumes using the existing spatial-then-depth filter sequence.
    fn interpolate3d(x: FloatTensor<B>, size: [usize; 3], options: InterpolateOptions) -> FloatTensor<B> {
        interpolation::interpolate3d_from_2d::<B>(x, size, options)
    }

    /// Backward volume resizing through the original spatial and depth filters.
    fn interpolate3d_backward(x: FloatTensor<B>, grad: FloatTensor<B>, size: [usize; 3],
        options: InterpolateOptions) -> FloatTensor<B> {
        interpolation::interpolate3d_backward_from_2d::<B>(x, grad, size, options)
    }

    /// Computes scaled dot-product attention: softmax(QKᵗ * scale) · V,
    /// where scale defaults to 1/sqrt(head_dim). Optionally applies masking,
    /// additive bias, causal masking, and softcap to the attention scores.
    ///
    /// # Arguments
    /// - `query`: Query tensor of shape `[batch_size, num_heads, seq_len_q, head_dim]`
    /// - `key`: Key tensor of shape `[batch_size, num_heads, seq_len_k, head_dim]`
    /// - `value`: Value tensor of shape `[batch_size, num_heads, seq_len_k, val_dim]`
    /// - `mask`: Optional boolean mask of shape `[batch_size, num_heads, seq_len_q, seq_len_k]`,
    ///   where `true` indicates positions to mask (i.e. set to -inf before softmax).
    /// - `attn_bias`: Optional float tensor of shape `[batch_size, num_heads, seq_len_q, seq_len_k]`
    ///   added to the attention scores before softmax (e.g. ALiBi, relative position biases).
    /// - `options`: Additional attention options (custom scale, softcap, causal masking).
    ///
    /// # Returns
    /// A tensor of shape `[batch_size, num_heads, seq_len_q, val_dim]`
    /// representing the attended context per head.
    ///
    /// # Note
    /// This implementation does not support dropout and is intended for inference or
    /// use cases where dropout is not needed.
    fn attention(
        query: FloatTensor<B>,
        key: FloatTensor<B>,
        value: FloatTensor<B>,
        mask: Option<BoolTensor<B>>,
        attn_bias: Option<FloatTensor<B>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<B>;

    /// Applies Layer Normalization over the last dimension of the input tensor.
    ///
    /// Computes `(x - mean) / sqrt(var + epsilon) * gamma + beta`, where `mean` and
    /// (biased) `var` are reduced over the last axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor of shape `[..., d_model]`.
    /// * `gamma` - Scale tensor of shape `[d_model]`.
    /// * `beta` - Optional bias tensor of shape `[d_model]`.
    /// * `epsilon` - Numerical stability term added to the variance before the square root.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor`.
    fn layer_norm(
        tensor: FloatTensor<B>,
        gamma: FloatTensor<B>,
        beta: Option<FloatTensor<B>>,
        epsilon: f64,
    ) -> FloatTensor<B> {
        Self::layer_norm_default(tensor, gamma, beta, epsilon)
    }

    /// Whether native forward statistics and complete first-order backward are available.
    fn has_layer_norm_backward() -> bool { false }

    /// Whether saved RMSNorm statistics and complete first-order backward are available.
    fn has_rms_norm_backward() -> bool { false }

    /// Last-axis RMSNorm with working arithmetic and one final output-storage cast.
    fn rms_norm(tensor: FloatTensor<B>, gamma: FloatTensor<B>, epsilon: f64) -> FloatTensor<B> {
        Self::rms_norm_with_stats(tensor, gamma, epsilon).output
    }

    /// RMSNorm forward retaining reciprocal row norms for backward.
    fn rms_norm_with_stats(tensor: FloatTensor<B>, gamma: FloatTensor<B>, epsilon: f64) -> RmsNormOutput<B> {
        super::normalization::rms_norm_with_stats::<B>(tensor, gamma, epsilon)
    }

    /// RMSNorm derivatives from saved reciprocal row norms, without repeating forward.
    fn rms_norm_backward(tensor: FloatTensor<B>, gamma: FloatTensor<B>, grad: FloatTensor<B>,
        rstd: FloatTensor<B>) -> RmsNormBackward<B> {
        super::normalization::rms_norm_backward::<B>(tensor, gamma, grad, rstd)
    }

    /// RMSNorm input and weight gradients selected by the two corresponding mask entries.
    fn rms_norm_backward_select(tensor: FloatTensor<B>, gamma: FloatTensor<B>, grad: FloatTensor<B>,
        rstd: FloatTensor<B>, mask: [bool; 2]) -> [Option<FloatTensor<B>>; 2] {
        super::normalization::rms_norm_backward_select::<B>(tensor, gamma, grad, rstd, mask)
    }

    /// Native forward with statistics retained for backward.
    fn layer_norm_with_stats(
        tensor: FloatTensor<B>, gamma: FloatTensor<B>,
        beta: Option<FloatTensor<B>>, epsilon: f64,
    ) -> LayerNormOutput<B> {
        super::normalization::layer_norm_with_stats::<B>(tensor, gamma, beta, epsilon)
    }

    /// Native backward using the exact statistics returned by forward.
    fn layer_norm_backward(
        tensor: FloatTensor<B>, gamma: FloatTensor<B>, grad: FloatTensor<B>,
        mean: FloatTensor<B>, rstd: FloatTensor<B>,
    ) -> LayerNormBackward<B> {
        super::normalization::layer_norm_backward::<B>(tensor, gamma, grad, mean, rstd)
    }

    /// LayerNorm input, weight and bias gradients selected in that order.
    fn layer_norm_backward_select(tensor: FloatTensor<B>, gamma: FloatTensor<B>, grad: FloatTensor<B>,
        mean: FloatTensor<B>, rstd: FloatTensor<B>, mask: [bool; 3]) -> [Option<FloatTensor<B>>; 3] {
        super::normalization::layer_norm_backward_select::<B>(tensor, gamma, grad, mean, rstd, mask)
    }

    /// Differentiable primitive composition for backends without native LayerNorm backward.
    fn layer_norm_default(
        tensor: FloatTensor<B>, gamma: FloatTensor<B>,
        beta: Option<FloatTensor<B>>, epsilon: f64,
    ) -> FloatTensor<B> {
        let shape = tensor.shape();
        let rank = shape.num_dims();
        let last_dim = rank - 1;
        let d_model = shape[last_dim];

        let mean = B::float_mean_dim(tensor.clone(), last_dim);
        let centered = B::float_sub(tensor, mean);
        let var = B::float_mean_dim(B::float_mul(centered.clone(), centered.clone()), last_dim);
        let denom = B::float_sqrt(B::float_add_scalar(var, epsilon.into()));
        let normalized = B::float_div(centered, denom);

        let broadcast_dims: alloc::vec::Vec<usize> = (0..rank)
            .map(|i| if i == last_dim { d_model } else { 1 })
            .collect();
        let gamma_b = B::float_reshape(gamma, Shape::from(broadcast_dims.clone()));
        let scaled = B::float_mul(normalized, gamma_b);

        match beta {
            Some(beta) => {
                let beta_b = B::float_reshape(beta, Shape::from(broadcast_dims));
                B::float_add(scaled, beta_b)
            }
            None => scaled,
        }
    }

    /// Computes the Connectionist Temporal Classification (CTC) loss.
    ///
    /// Sums over all valid alignments between the input and target sequences
    /// using the forward (alpha) algorithm.
    ///
    /// # Arguments
    ///
    /// * `log_probs` - Log-probabilities of shape `[T, N, C]`
    /// * `targets` - Target label indices of shape `[N, S]`
    /// * `input_lengths` - Actual input sequence lengths per batch element `[N]`
    /// * `target_lengths` - Actual target lengths per batch element `[N]`
    /// * `blank` - Index of the blank label
    ///
    /// # Returns
    ///
    /// Per-sample loss of shape `[N]`
    fn ctc_loss(
        log_probs: FloatTensor<B>,
        targets: IntTensor<B>,
        input_lengths: IntTensor<B>,
        target_lengths: IntTensor<B>,
        blank: usize,
    ) -> FloatTensor<B> {
        ctc::ctc_loss_default::<B>(log_probs, targets, input_lengths, target_lengths, blank)
    }

    /// Returns `true` if this backend implements [ctc_loss_backward](ModuleOps::ctc_loss_backward)
    /// natively.
    ///
    /// Autodiff queries this flag to decide between two paths:
    /// - `true`: use the backend's [ctc_loss](ModuleOps::ctc_loss) and
    ///   [ctc_loss_backward](ModuleOps::ctc_loss_backward) directly.
    /// - `false`: call [ctc::ctc_loss_default] for the forward pass; autodiff
    ///   then differentiates through the decomposed tensor ops.
    ///
    /// Backends that override `ctc_loss_backward` must also override this to
    /// return `true`.
    fn has_ctc_loss_backward() -> bool {
        false
    }

    /// Backward pass for [ctc_loss](ModuleOps::ctc_loss): gradient w.r.t. `log_probs`.
    ///
    /// Only called when [has_ctc_loss_backward](ModuleOps::has_ctc_loss_backward)
    /// returns `true`. Backends without a native implementation should leave
    /// both methods at their defaults; the gradient is computed automatically by
    /// autodiff against the decomposed [ctc::ctc_loss_default] forward.
    ///
    /// # Arguments
    ///
    /// * `log_probs` - Log-probabilities of shape `[T, N, C]`
    /// * `targets` - Target label indices of shape `[N, S]`
    /// * `input_lengths` - Actual input sequence lengths per batch element `[N]`
    /// * `target_lengths` - Actual target lengths per batch element `[N]`
    /// * `grad_loss` - Upstream gradient w.r.t. the per-sample loss `[N]`
    /// * `blank` - Index of the blank label
    ///
    /// # Returns
    ///
    /// Gradient w.r.t. `log_probs` of shape `[T, N, C]`
    fn ctc_loss_backward(
        _log_probs: FloatTensor<B>,
        _targets: IntTensor<B>,
        _input_lengths: IntTensor<B>,
        _target_lengths: IntTensor<B>,
        _grad_loss: FloatTensor<B>,
        _blank: usize,
    ) -> FloatTensor<B> {
        unreachable!(
            "ctc_loss_backward called on a backend whose has_ctc_loss_backward() returns false"
        )
    }

    /// Real-valued FFT with optional size parameter.
    ///
    /// When `n` is `None`, the signal must be a power of two along `dim`, and the output has
    /// `signal_len / 2 + 1` frequency bins.
    ///
    /// When `n` is `Some(size)`, `size` must also be a power of two. The signal is truncated
    /// or zero-padded to `size` and the output has `size / 2 + 1` frequency bins. Non-power-
    /// of-two sizes are currently rejected at the public API boundary; true arbitrary-`n` DFT
    /// support (Bluestein's algorithm) is tracked as a follow-up.
    ///
    /// Returns two tensors: the real part and the imaginary part.
    fn rfft(
        signal: FloatTensor<B>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<B>, FloatTensor<B>);

    /// Inverse real-valued FFT with optional output size.
    ///
    /// When `n` is `None`, the reconstructed signal length `2 * (spectrum_size - 1)` must be
    /// a power of two.
    ///
    /// When `n` is `Some(size)`, `size` must also be a power of two. Output has exactly
    /// `size` samples.
    fn irfft(
        spectrum_re: FloatTensor<B>,
        spectrum_im: FloatTensor<B>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<B>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic = "stride must be non-zero"]
    fn conv_options_stride_zero() {
        let _opt = ConvOptions::new([0, 1], [0, 0], [1, 1], 1);
    }

    #[test]
    #[should_panic = "dilation must be non-zero"]
    fn conv_options_dilation_zero() {
        let _opt = ConvOptions::new([1, 1], [0, 0], [0, 0], 1);
    }

    #[test]
    #[should_panic = "groups must be non-zero"]
    fn conv_options_groups_zero() {
        let _opt = ConvOptions::new([1, 1], [0, 0], [1, 1], 0);
    }

    #[test]
    #[should_panic = "stride must be non-zero"]
    fn conv_transpose_options_stride_zero() {
        let _opt = ConvTransposeOptions::new([0, 1], [0, 0], [0, 0], [1, 1], 1);
    }

    #[test]
    #[should_panic = "dilation must be non-zero"]
    fn conv_transpose_options_dilation_zero() {
        let _opt = ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [0, 0], 1);
    }

    #[test]
    #[should_panic = "groups must be non-zero"]
    fn conv_transpose_options_groups_zero() {
        let _opt = ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [1, 1], 0);
    }

    #[test]
    #[should_panic = "stride must be non-zero"]
    fn deform_conv_options_stride_zero() {
        let _opt = DeformConvOptions::new([0, 1], [0, 0], [1, 1], 1, 1);
    }

    #[test]
    #[should_panic = "dilation must be non-zero"]
    fn deform_conv_options_dilation_zero() {
        let _opt = DeformConvOptions::new([1, 1], [0, 0], [0, 0], 1, 1);
    }

    #[test]
    #[should_panic = "weight groups must be non-zero"]
    fn deform_conv_options_weights_groups_zero() {
        let _opt = DeformConvOptions::new([1, 1], [0, 0], [1, 1], 0, 1);
    }

    #[test]
    #[should_panic = "offset groups must be non-zero"]
    fn deform_conv_options_offset_groups_zero() {
        let _opt = DeformConvOptions::new([1, 1], [0, 0], [1, 1], 1, 0);
    }

    #[test]
    #[should_panic = "stride must be non-zero"]
    fn unfold_options_stride_zero() {
        let _opt = UnfoldOptions::new([0, 1], [0, 0], [1, 1]);
    }

    #[test]
    #[should_panic = "dilation must be non-zero"]
    fn unfold_options_dilation_zero() {
        let _opt = UnfoldOptions::new([1, 1], [0, 0], [0, 0]);
    }
}
