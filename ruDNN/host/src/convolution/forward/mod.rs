//! Forward convolution operations using tiled im2col + gemm approach.
//!
//! All convolutions (1D, 2D, 3D) use a unified 3D implementation:
//! - conv1d: adds two size-1 dimensions, calls conv3d, squeezes output
//! - conv2d: adds one size-1 dimension, calls conv3d, squeezes output
//! - conv3d: native implementation
//!
//! Optimizations:
//! - Tiled im2col: Process output in tiles for better cache usage and parallelism
//! - NHWC layout: Convert to channels-last for cache-friendly access
//! - Nested parallelism: Batch and tile dimensions run in parallel via rayon
//! - 1x1 fast path: Skip im2col for pointwise convolutions
//! - Depthwise fast path: For canonical depthwise (groups == c_in == c_out,
//!   channels_per_group == 1), skip NHWC conversion, im2col, and gemm entirely.
//!   Uses a direct per-(b, c) accumulate with analytic bounds so the inner
//!   spatial loop has no padding checks and autovectorizes.
//! - Small-channel fast path: For groups=1 convs with very few input channels
//!   (e.g. 3-channel Sobel-style edge filters), reuse the depthwise kernel by
//!   accumulating over input channels. Skips the same NHWC/im2col/gemm overhead
//!   as the depthwise path. Wins when `channels_in` is small enough that
//!   gemm's per-dispatch setup cost dominates over the tiny inner compute.
//! - Direct conv path: For small-spatial 1D-like convolutions, decompose into
//!   per-kernel-position gemm calls on NCHW data, skipping NHWC conversion and im2col
//!
//! Supported dtypes: f32, f64, f16 (native gemm), bf16 (via f32 conversion)

use alloc::vec;
use alloc::vec::Vec;
use ruda_core::tensor::DType;
use ruda_core::tensor::spatial::ConvOptions;
use ruda_core::tensor::spatial::calculate_conv_output_size;
use ruda_core::{bytes::Bytes, tensor::Shape};
use half::f16;

use ruda_core::tensor::host::{HostTensor, Layout};

use super::common::{add_bias, squeeze_3d_to_1d, squeeze_3d_to_2d};

// ============================================================================
// Macros for forward conv
// ============================================================================

/// Generates a conv3d_1x1 function that uses the optimized gemm fast path.
macro_rules! conv3d_1x1_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $one:expr, $add_fn:expr) => {
        pub(super) fn $fn_name(
            x: HostTensor,
            weight: HostTensor,
            bias: Option<HostTensor>,
            options: &ConvOptions<3>,
        ) -> HostTensor {
            conv3d_1x1_impl::<$T>(x, weight, bias, options, $dtype, $zero, $one, $add_fn)
        }
    };
}

/// Generates a conv3d typed function with 1x1, depthwise, small-channel, and
/// direct fast-path checks.
macro_rules! conv3d_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $gemm_fn:ident, $add_fn:expr, $fn_1x1:ident, $fn_depthwise:ident, $fn_small_channel:ident $(, $fn_direct:ident)?) => {
        pub fn $fn_name(
            x: HostTensor,
            weight: HostTensor,
            bias: Option<HostTensor>,
            options: &ConvOptions<3>,
        ) -> HostTensor {
            let w_shape = weight.layout().shape();
            if is_1x1_conv(w_shape[2], w_shape[3], w_shape[4], options) {
                return $fn_1x1(x, weight, bias, options);
            }
            let x_shape = x.layout().shape();
            if should_use_depthwise_conv(x_shape, w_shape, options) {
                return $fn_depthwise(x, weight, bias, options);
            }
            if should_use_small_channel_conv(x_shape, w_shape, options) {
                return $fn_small_channel(x, weight, bias, options);
            }
            $(
                if should_use_direct_conv(x_shape, w_shape, options) {
                    return $fn_direct(x, weight, bias, options);
                }
            )?
            conv3d_impl::<$T>(x, weight, bias, options, $dtype, $zero, $gemm_fn, $add_fn)
        }
    };
}

// ============================================================================
// Conv1d - delegates to conv3d
// ============================================================================

conv_nd_via_3d!(
    conv1d_f32,
    conv3d_f32,
    expand_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvOptions
);
conv_nd_via_3d!(
    conv1d_f64,
    conv3d_f64,
    expand_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvOptions
);
conv_nd_via_3d!(
    conv1d_f16,
    conv3d_f16,
    expand_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvOptions
);
bf16_via_f32!(conv1d_bf16, conv1d_f32, 1, ConvOptions);

fn expand_1d_to_3d(
    x: &HostTensor,
    weight: &HostTensor,
    options: &ConvOptions<1>,
) -> (HostTensor, HostTensor, ConvOptions<3>) {
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![x_shape[0], x_shape[1], 1, 1, x_shape[2]]));

    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![w_shape[0], w_shape[1], 1, 1, w_shape[2]]));

    let options_3d = ConvOptions::new(
        [1, 1, options.stride[0]],
        [0, 0, options.padding[0]],
        [1, 1, options.dilation[0]],
        options.groups,
    );

    (x_3d, weight_3d, options_3d)
}

// ============================================================================
// Conv2d - delegates to conv3d
// ============================================================================

conv_nd_via_3d!(
    conv2d_f32,
    conv3d_f32,
    expand_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvOptions
);
conv_nd_via_3d!(
    conv2d_f64,
    conv3d_f64,
    expand_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvOptions
);
conv_nd_via_3d!(
    conv2d_f16,
    conv3d_f16,
    expand_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvOptions
);
bf16_via_f32!(conv2d_bf16, conv2d_f32, 2, ConvOptions);

fn expand_2d_to_3d(
    x: &HostTensor,
    weight: &HostTensor,
    options: &ConvOptions<2>,
) -> (HostTensor, HostTensor, ConvOptions<3>) {
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![
        x_shape[0], x_shape[1], 1, x_shape[2], x_shape[3],
    ]));

    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![
        w_shape[0], w_shape[1], 1, w_shape[2], w_shape[3],
    ]));

    let options_3d = ConvOptions::new(
        [1, options.stride[0], options.stride[1]],
        [0, options.padding[0], options.padding[1]],
        [1, options.dilation[0], options.dilation[1]],
        options.groups,
    );

    (x_3d, weight_3d, options_3d)
}

// ============================================================================
// Conv3d - native implementations
// ============================================================================

conv3d_typed!(
    conv3d_f32,
    f32,
    DType::F32,
    0.0f32,
    gemm_f32,
    |a, b| a + b,
    conv3d_1x1_f32,
    conv3d_depthwise_f32,
    conv3d_small_channel_f32,
    conv3d_direct_f32
);
conv3d_typed!(
    conv3d_f64,
    f64,
    DType::F64,
    0.0f64,
    gemm_f64,
    |a, b| a + b,
    conv3d_1x1_f64,
    conv3d_depthwise_f64,
    conv3d_small_channel_f64,
    conv3d_direct_f64
);
conv3d_typed!(
    conv3d_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    gemm_f16,
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32()),
    conv3d_1x1_f16,
    conv3d_depthwise_f16,
    conv3d_small_channel_f16
);
bf16_via_f32!(conv3d_bf16, conv3d_f32, 3, ConvOptions);

mod im2col;
use im2col::*;

mod pointwise;
use pointwise::*;

mod depthwise;
use depthwise::*;

mod small_channel;
use small_channel::*;

mod direct;
use direct::*;

mod gemm_dispatch;
use gemm_dispatch::*;

// ============================================================================
// Tests
// ============================================================================

// Tests kept here exercise flex-specific dtype storage paths (f64/f16/bf16)
// and flex-internal fast-path dispatch (1x1, depthwise, small-channel,
// direct, oh-outer) plus validation/panic checks on flex helpers. Plain
// conv shape/stride/padding/groups correctness is covered at the public
// API level by crates/ruda-backend-tests/tests/tensor/float/module/
// conv{1,2,3}d.rs, so those tests were removed from here. When adding
// new tests, keep them here only if they probe flex dtype dispatch or a
// flex-internal fast path; otherwise add them there.
#[cfg(test)]
mod tests;
