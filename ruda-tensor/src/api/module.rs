use crate::api::{
    Bool, Int, Tensor, TensorPrimitive,
    backend::Backend,
    check,
    check::TensorCheck,
    ops::{
        AttentionModuleOptions, ConvOptions, ConvTransposeOptions, InterpolateOptions, PadMode,
        PaddedConvOptions, UnfoldOptions,
    },
};

use super::ops::DeformConvOptions;

/// Computes the [CTC loss](crate::api::ops::ModuleOps::ctc_loss).
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
pub fn ctc_loss<B>(
    log_probs: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    input_lengths: Tensor<B, 1, Int>,
    target_lengths: Tensor<B, 1, Int>,
    blank: usize,
) -> Tensor<B, 1>
where
    B: Backend,
{
    Tensor::new(TensorPrimitive::Float(B::ctc_loss(
        log_probs.primitive.tensor(),
        targets.primitive,
        input_lengths.primitive,
        target_lengths.primitive,
        blank,
    )))
}

/// Applies the [embedding module](crate::api::ops::ModuleOps::embedding).
pub fn embedding<B>(weights: Tensor<B, 2>, indices: Tensor<B, 2, Int>) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(TensorPrimitive::Float(B::embedding(
        weights.primitive.tensor(),
        indices.primitive,
    )))
}

/// Applies a [1D convolution](crate::api::ops::ModuleOps::conv1d).
///
/// Accepts [`ConvOptions`] for symmetric padding, or [`PaddedConvOptions`] for
/// asymmetric padding. When asymmetric padding is specified, an explicit pad
/// operation is applied before the convolution backend op.
pub fn conv1d<B>(
    x: Tensor<B, 3>,
    weight: Tensor<B, 3>,
    bias: Option<Tensor<B, 1>>,
    options: impl Into<PaddedConvOptions<1>>,
) -> Tensor<B, 3>
where
    B: Backend,
{
    let padded_options = options.into();
    check!(TensorCheck::conv(
        "conv1d",
        x.dims(),
        weight.dims(),
        padded_options.options.groups,
    ));

    if let Some(padding_end) = padded_options.padding_end {
        let left = padded_options.options.padding[0];
        let right = padding_end[0];
        // For 1D (NCL format), pad the length dimension
        let padded = x.pad((left, right, 0, 0), PadMode::Constant(0.0));
        let zero_options = ConvOptions::new(
            padded_options.options.stride,
            [0],
            padded_options.options.dilation,
            padded_options.options.groups,
        );
        Tensor::new(TensorPrimitive::Float(B::conv1d(
            padded.primitive.tensor(),
            weight.primitive.tensor(),
            bias.map(|b| b.primitive.tensor()),
            zero_options,
        )))
    } else {
        Tensor::new(TensorPrimitive::Float(B::conv1d(
            x.primitive.tensor(),
            weight.primitive.tensor(),
            bias.map(|b| b.primitive.tensor()),
            padded_options.options,
        )))
    }
}

/// Applies a [2D convolution](crate::api::ops::ModuleOps::conv2d).
///
/// Accepts [`ConvOptions`] for symmetric padding, or [`PaddedConvOptions`] for
/// asymmetric padding. When asymmetric padding is specified, an explicit pad
/// operation is applied before the convolution backend op.
pub fn conv2d<B>(
    x: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    options: impl Into<PaddedConvOptions<2>>,
) -> Tensor<B, 4>
where
    B: Backend,
{
    let padded_options = options.into();
    check!(TensorCheck::conv(
        "conv2d",
        x.dims(),
        weight.dims(),
        padded_options.options.groups,
    ));

    if let Some(padding_end) = padded_options.padding_end {
        let top = padded_options.options.padding[0];
        let left = padded_options.options.padding[1];
        let bottom = padding_end[0];
        let right = padding_end[1];
        // For 2D (NCHW format), pad height and width
        let padded = x.pad((left, right, top, bottom), PadMode::Constant(0.0));
        let zero_options = ConvOptions::new(
            padded_options.options.stride,
            [0, 0],
            padded_options.options.dilation,
            padded_options.options.groups,
        );
        Tensor::new(TensorPrimitive::Float(B::conv2d(
            padded.primitive.tensor(),
            weight.primitive.tensor(),
            bias.map(|b| b.primitive.tensor()),
            zero_options,
        )))
    } else {
        Tensor::new(TensorPrimitive::Float(B::conv2d(
            x.primitive.tensor(),
            weight.primitive.tensor(),
            bias.map(|b| b.primitive.tensor()),
            padded_options.options,
        )))
    }
}

/// Applies a [3D convolution](crate::api::ops::ModuleOps::conv3d).
///
/// Accepts [`ConvOptions`] for symmetric padding, or [`PaddedConvOptions`] for
/// asymmetric padding. An explicit pad operation handles asymmetric padding
/// before dispatching to the convolution backend.
pub fn conv3d<B>(
    x: Tensor<B, 5>,
    weight: Tensor<B, 5>,
    bias: Option<Tensor<B, 1>>,
    options: impl Into<PaddedConvOptions<3>>,
) -> Tensor<B, 5>
where
    B: Backend,
{
    let padded_options = options.into();
    check!(TensorCheck::conv(
        "conv3d",
        x.dims(),
        weight.dims(),
        padded_options.options.groups,
    ));

    let mut options = padded_options.options;
    let x = if let Some(padding_end) = padded_options.padding_end {
        let padding: [(usize, usize); 3] =
            core::array::from_fn(|axis| (options.padding[axis], padding_end[axis]));
        options.padding = [0; 3];
        x.pad(padding, PadMode::Constant(0.0))
    } else {
        x
    };

    Tensor::new(TensorPrimitive::Float(B::conv3d(
        x.primitive.tensor(),
        weight.primitive.tensor(),
        bias.map(|b| b.primitive.tensor()),
        options,
    )))
}

/// Applies a [Deformable 2D convolution](crate::api::ops::ModuleOps::deform_conv2d).
pub fn deform_conv2d<B>(
    x: Tensor<B, 4>,
    offset: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    mask: Option<Tensor<B, 4>>,
    bias: Option<Tensor<B, 1>>,
    options: DeformConvOptions<2>,
) -> Tensor<B, 4>
where
    B: Backend,
{
    check!(TensorCheck::conv(
        "deform_conv2d",
        x.dims(),
        weight.dims(),
        options.weight_groups,
    ));
    let [batch, channels, height, width] = x.dims();
    let [_, _, kernel_height, kernel_width] = weight.dims();
    assert!(
        options.offset_groups > 0 && channels.is_multiple_of(options.offset_groups),
        "deform_conv2d input channels must be divisible by non-zero offset groups"
    );
    let [out_height, out_width] = options.output_size(
        [height, width], [kernel_height, kernel_width],
    );
    let mask_channels = options.offset_groups * kernel_height * kernel_width;
    assert_eq!(
        offset.dims(), [batch, 2 * mask_channels, out_height, out_width],
        "deform_conv2d offset shape must match groups, kernel and output"
    );
    if let Some(mask) = mask.as_ref() {
        assert_eq!(
            mask.dims(), [batch, mask_channels, out_height, out_width],
            "deform_conv2d mask shape must match groups, kernel and output"
        );
    }
    Tensor::new(TensorPrimitive::Float(B::deform_conv2d(
        x.primitive.tensor(),
        offset.primitive.tensor(),
        weight.primitive.tensor(),
        mask.map(|m| m.primitive.tensor()),
        bias.map(|b| b.primitive.tensor()),
        options,
    )))
}

/// Applies a [1D transposed convolution](crate::api::ops::ModuleOps::conv_transpose1d).
pub fn conv_transpose1d<B>(
    x: Tensor<B, 3>,
    weight: Tensor<B, 3>,
    bias: Option<Tensor<B, 1>>,
    options: ConvTransposeOptions<1>,
) -> Tensor<B, 3>
where
    B: Backend,
{
    check!(TensorCheck::conv_transpose(
        "conv_transpose1d",
        x.dims(),
        weight.dims(),
    ));
    Tensor::new(TensorPrimitive::Float(B::conv_transpose1d(
        x.primitive.tensor(),
        weight.primitive.tensor(),
        bias.map(|b| b.primitive.tensor()),
        options,
    )))
}

/// Applies a [2D transposed convolution](crate::api::ops::ModuleOps::conv_transpose2d).
pub fn conv_transpose2d<B>(
    x: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    options: ConvTransposeOptions<2>,
) -> Tensor<B, 4>
where
    B: Backend,
{
    check!(TensorCheck::conv_transpose(
        "conv_transpose2d",
        x.dims(),
        weight.dims(),
    ));
    Tensor::new(TensorPrimitive::Float(B::conv_transpose2d(
        x.primitive.tensor(),
        weight.primitive.tensor(),
        bias.map(|b| b.primitive.tensor()),
        options,
    )))
}

/// Applies a 3D transposed convolution](crate::api::ops::ModuleOps::conv_transpose3d).
pub fn conv_transpose3d<B>(
    x: Tensor<B, 5>,
    weight: Tensor<B, 5>,
    bias: Option<Tensor<B, 1>>,
    options: ConvTransposeOptions<3>,
) -> Tensor<B, 5>
where
    B: Backend,
{
    check!(TensorCheck::conv_transpose(
        "conv_transpose3d",
        x.dims(),
        weight.dims(),
    ));
    Tensor::new(TensorPrimitive::Float(B::conv_transpose3d(
        x.primitive.tensor(),
        weight.primitive.tensor(),
        bias.map(|b| b.primitive.tensor()),
        options,
    )))
}

/// Applies a [4D to 3D unfold](crate::api::ops::ModuleOps::unfold4d).
pub fn unfold4d<B>(x: Tensor<B, 4>, kernel_size: [usize; 2], options: UnfoldOptions) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(TensorPrimitive::Float(B::unfold4d(
        x.primitive.tensor(),
        kernel_size,
        options,
    )))
}

/// Applies a [1D max pooling](crate::api::ops::ModuleOps::max_pool1d).
pub fn max_pool1d<B>(
    x: Tensor<B, 3>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(TensorPrimitive::Float(B::max_pool1d(
        x.primitive.tensor(),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )))
}

/// Applies a [2D max pooling](crate::api::ops::ModuleOps::max_pool2d).
pub fn max_pool2d<B>(
    x: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> Tensor<B, 4>
where
    B: Backend,
{
    Tensor::new(TensorPrimitive::Float(B::max_pool2d(
        x.primitive.tensor(),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )))
}

/// Applies a [2D avg pooling](crate::api::ops::ModuleOps::avg_pool2d).
pub fn avg_pool2d<B>(
    x: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> Tensor<B, 4>
where
    B: Backend,
{
    Tensor::new(TensorPrimitive::Float(B::avg_pool2d(
        x.primitive.tensor(),
        kernel_size,
        stride,
        padding,
        count_include_pad,
        ceil_mode,
    )))
}

/// Applies a [1D avg pooling](crate::api::ops::ModuleOps::avg_pool1d).
pub fn avg_pool1d<B>(
    x: Tensor<B, 3>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
    ceil_mode: bool,
) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(TensorPrimitive::Float(B::avg_pool1d(
        x.primitive.tensor(),
        kernel_size,
        stride,
        padding,
        count_include_pad,
        ceil_mode,
    )))
}

/// Applies a [1D max pooling](crate::api::ops::ModuleOps::max_pool1d).
pub fn max_pool1d_with_indices<B>(
    x: Tensor<B, 3>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> (Tensor<B, 3>, Tensor<B, 3, Int>)
where
    B: Backend,
{
    let output = B::max_pool1d_with_indices(
        x.primitive.tensor(),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    );

    (
        Tensor::new(TensorPrimitive::Float(output.output)),
        Tensor::new(output.indices),
    )
}

/// Applies a [2D max pooling with indices](crate::api::ops::ModuleOps::max_pool2d_with_indices).
pub fn max_pool2d_with_indices<B>(
    x: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> (Tensor<B, 4>, Tensor<B, 4, Int>)
where
    B: Backend,
{
    let output = B::max_pool2d_with_indices(
        x.primitive.tensor(),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    );

    (
        Tensor::new(TensorPrimitive::Float(output.output)),
        Tensor::new(output.indices),
    )
}

/// Applies a [2D adaptive avg pooling](crate::api::ops::ModuleOps::adaptive_avg_pool2d).
pub fn adaptive_avg_pool2d<B>(x: Tensor<B, 4>, output_size: [usize; 2]) -> Tensor<B, 4>
where
    B: Backend,
{
    Tensor::new(TensorPrimitive::Float(B::adaptive_avg_pool2d(
        x.primitive.tensor(),
        output_size,
    )))
}

/// Applies a [1D adaptive avg pooling](crate::api::ops::ModuleOps::adaptive_avg_pool1d).
pub fn adaptive_avg_pool1d<B>(x: Tensor<B, 3>, output_size: usize) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(TensorPrimitive::Float(B::adaptive_avg_pool1d(
        x.primitive.tensor(),
        output_size,
    )))
}

/// Applies a [2D interpolation](crate::api::ops::ModuleOps::interpolate).
pub fn interpolate<B>(
    x: Tensor<B, 4>,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> Tensor<B, 4>
where
    B: Backend,
{
    Tensor::new(TensorPrimitive::Float(B::interpolate(
        x.primitive.tensor(),
        output_size,
        options,
    )))
}

/// Applies a linear transformation to the input tensor using the given weight and bias.
///
/// ```math
/// y = x @ weight + [bias]
/// ```
///
/// # Arguments:
///
/// - `input` is the input tensor, ``[..., d_input]``.
/// - `weight` is the weight tensor, ``[d_input, d_output]``.
/// - `bias` is the bias tensor (optional), ``[d_output]``.
///
/// # Returns:
///
/// The transformed tensor, ``[..., d_output]``.
///
/// # Compatibility
///
/// This function differs from PyTorch's ``torch.nn.functional.linear`` in that it does not
/// transpose the weight matrix. In PyTorch, the weight matrix is transposed before
/// multiplication:
///
/// ```math
/// y = x @ weight^T + [bias]
/// ```
pub fn linear<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
) -> Tensor<B, D> {
    if D == 1 {
        // Insert and remove an extra batch dimension for the batch matmul to work.
        let input = input.unsqueeze::<2>();
        let output = linear(input, weight, bias);
        return output.squeeze_dim(0);
    }

    Tensor::new(TensorPrimitive::Float(B::linear(
        input.primitive.tensor(),
        weight.primitive.tensor(),
        bias.map(|b| b.primitive.tensor()),
    )))
}

/// Computes scaled dot-product attention: softmax(QKᵗ * scale) · V,
/// where scale defaults to 1/sqrt(head_dim) (configurable via `options.scale`).
/// Optionally applies masking, additive bias, causal masking, and softcap.
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
pub fn attention<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
    attn_bias: Option<Tensor<B, 4>>,
    options: AttentionModuleOptions,
) -> Tensor<B, 4> {
    Tensor::new(TensorPrimitive::Float(B::attention(
        query.primitive.tensor(),
        key.primitive.tensor(),
        value.primitive.tensor(),
        mask.map(|mask| mask.primitive),
        attn_bias.map(|bias| bias.primitive.tensor()),
        options,
    )))
}

/// Exports attention fallback to test backend's attention against.
pub fn attention_fallback<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
    attn_bias: Option<Tensor<B, 4>>,
    options: AttentionModuleOptions,
) -> Tensor<B, 4> {
    Tensor::new(TensorPrimitive::Float(
        crate::api::ops::attention::attention_fallback::<B>(
            query.primitive.tensor(),
            key.primitive.tensor(),
            value.primitive.tensor(),
            mask.map(|mask| mask.primitive),
            attn_bias.map(|bias| bias.primitive.tensor()),
            options,
        ),
    ))
}
