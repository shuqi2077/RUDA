use super::{BasicOps, DType, Int, Tensor, backend::Backend};
use super::ops::PadMode;
use super::module::{
    adaptive_avg_pool1d, adaptive_avg_pool2d, avg_pool1d, avg_pool2d,
    max_pool1d, max_pool2d, max_pool1d_with_indices, max_pool2d_with_indices,
};

pub(super) fn volume_planes<B: Backend, K: BasicOps<B>>(input: Tensor<B, 5, K>) -> Tensor<B, 4, K> {
    let [batch, channels, depth, height, width] = input.dims();
    let planes = batch.checked_mul(depth).expect("pooling plane count overflow");
    input.permute([0, 2, 1, 3, 4]).reshape([planes, channels, height, width])
}

pub(super) fn plane_depth_lines<B: Backend, K: BasicOps<B>>(
    planes: Tensor<B, 4, K>,
    batch: usize,
    depth: usize,
) -> Tensor<B, 3, K> {
    let [_, channels, height, width] = planes.dims();
    let lines = batch.checked_mul(channels)
        .and_then(|count| count.checked_mul(height))
        .and_then(|count| count.checked_mul(width))
        .expect("pooling depth line count overflow");
    planes.reshape([batch, depth, channels, height, width])
        .permute([0, 2, 3, 4, 1]).reshape([lines, 1, depth])
}

pub(super) fn depth_lines_volume<B: Backend, K: BasicOps<B>>(
    lines: Tensor<B, 3, K>,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Tensor<B, 5, K> {
    let [_, _, depth] = lines.dims();
    lines.reshape([batch, channels, height, width, depth]).permute([0, 1, 4, 2, 3])
}

/// Pool native `[batch, channels, depth, height, width]` activations by maximum.
///
/// Executes the existing spatial and depth pooling kernels on the tensor's
/// backend. Padding, dilation and ceil mode are applied independently per axis.
/// Layout transformations and both pooling stages remain differentiable.
pub fn max_pool3d<B: Backend>(
    input: Tensor<B, 5>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> Tensor<B, 5> {
    let [batch, channels, depth, _, _] = input.dims();
    let planes = max_pool2d(
        volume_planes(input),
        [kernel_size[1], kernel_size[2]],
        [stride[1], stride[2]],
        [padding[1], padding[2]],
        [dilation[1], dilation[2]],
        ceil_mode,
    );
    let [_, _, height, width] = planes.dims();
    let lines = max_pool1d(
        plane_depth_lines(planes, batch, depth),
        kernel_size[0], stride[0], padding[0], dilation[0], ceil_mode,
    );
    depth_lines_volume(lines, batch, channels, height, width)
}

/// Maximum volume pooling with original input positions in native I64 storage.
///
/// Indices flatten each input volume as `d * (H * W) + h * W + w`, independently
/// for every batch/channel. Tie selection follows the
/// underlying spatial kernel followed by the depth kernel. Invalid backend
/// indices become `-1`; they are never used as out-of-bounds gather addresses.
pub fn max_pool3d_with_indices<B: Backend>(
    input: Tensor<B, 5>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> (Tensor<B, 5>, Tensor<B, 5, Int>) {
    let [batch, channels, depth, input_height, input_width] = input.dims();
    assert!(depth > 0 && input_height > 0 && input_width > 0,
        "volume pooling indices require non-empty spatial axes");
    let area = input_height.checked_mul(input_width).expect("pooling plane area overflow");
    let volume = depth.checked_mul(area).expect("pooling volume size overflow");
    assert!(volume <= i64::MAX as usize, "volume pooling indices exceed I64");
    let (planes, plane_indices) = max_pool2d_with_indices(
        volume_planes(input),
        [kernel_size[1], kernel_size[2]],
        [stride[1], stride[2]],
        [padding[1], padding[2]],
        [dilation[1], dilation[2]],
        ceil_mode,
    );
    let [_, _, height, width] = planes.dims();
    let (lines, depth_indices) = max_pool1d_with_indices(
        plane_depth_lines(planes, batch, depth),
        kernel_size[0], stride[0], padding[0], dilation[0], ceil_mode,
    );
    let depth_indices = depth_indices.cast(DType::I64);
    let safe_depth_indices = depth_indices.clone().clamp(0, depth as i64 - 1);
    let spatial_indices = plane_depth_lines(plane_indices.cast(DType::I64), batch, depth)
        .gather(2, safe_depth_indices.clone());
    let invalid = depth_indices.clone().lower_elem(0)
        .bool_or(depth_indices.clone().greater_equal_elem(depth as i64))
        .bool_or(spatial_indices.clone().lower_elem(0))
        .bool_or(spatial_indices.clone().greater_equal_elem(area as i64));
    let indices = (safe_depth_indices.mul_scalar(area as i64)
        + spatial_indices.clamp(0, area as i64 - 1)).mask_fill(invalid, -1);
    (
        depth_lines_volume(lines, batch, channels, height, width),
        depth_lines_volume(indices, batch, channels, height, width),
    )
}

/// Average-pool native `[batch, channels, depth, height, width]` activations.
///
/// The rectangular reduction is factored into native spatial and depth pooling.
/// Each stage retains the input's floating storage and its backend's arithmetic.
/// Padding divisors factor per axis, including partially covered ceil windows.
pub fn avg_pool3d<B: Backend>(
    input: Tensor<B, 5>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    count_include_pad: bool,
    ceil_mode: bool,
) -> Tensor<B, 5> {
    let [batch, channels, depth, _, _] = input.dims();
    let planes = avg_pool2d(
        volume_planes(input),
        [kernel_size[1], kernel_size[2]],
        [stride[1], stride[2]],
        [padding[1], padding[2]],
        count_include_pad,
        ceil_mode,
    );
    let [_, _, height, width] = planes.dims();
    let lines = avg_pool1d(
        plane_depth_lines(planes, batch, depth),
        kernel_size[0], stride[0], padding[0], count_include_pad, ceil_mode,
    );
    depth_lines_volume(lines, batch, channels, height, width)
}

/// Adaptive average pooling to explicit `[depth, height, width]` extents.
///
/// Native adaptive pooling bins are retained, including overlapping bins and
/// output extents larger than the input. No host reduction or interpolation is
/// used, and gradients flow through the original pooling and layout operations.
pub fn adaptive_avg_pool3d<B: Backend>(
    input: Tensor<B, 5>,
    output_size: [usize; 3],
) -> Tensor<B, 5> {
    let [batch, channels, depth, _, _] = input.dims();
    let planes = adaptive_avg_pool2d(volume_planes(input), [output_size[1], output_size[2]]);
    let [_, _, height, width] = planes.dims();
    let lines = adaptive_avg_pool1d(plane_depth_lines(planes, batch, depth), output_size[0]);
    depth_lines_volume(lines, batch, channels, height, width)
}

fn average_excluding_explicit_padding<B: Backend, const D: usize, const N: usize>(
    input: Tensor<B, D>,
    padding: [(usize, usize); N],
    pool: impl Fn(Tensor<B, D>) -> Tensor<B, D>,
) -> Tensor<B, D> {
    let storage = input.dtype();
    let compute = if storage == DType::F64 { DType::F64 } else { DType::F32 };
    let mut visible_shape = input.dims();
    visible_shape[0] = 1;
    visible_shape[1] = 1;
    let visible = Tensor::<B, D>::ones(visible_shape, (&input.device(), compute))
        .pad(padding, PadMode::Constant(0.0));
    let values = pool(input.cast(compute).pad(padding, PadMode::Constant(0.0)));
    let coverage = pool(visible);
    (values / coverage).cast(storage)
}

/// Average pooling with explicit `(left, right)` padding.
///
/// When padding is excluded, the denominator counts original input elements,
/// not the zeros materialized for asymmetric padding. Symmetric calls retain
/// the original backend operation and arithmetic.
pub fn avg_pool1d_padded<B: Backend>(
    input: Tensor<B, 3>,
    kernel_size: usize,
    stride: usize,
    padding: [(usize, usize); 1],
    count_include_pad: bool,
    ceil_mode: bool,
) -> Tensor<B, 3> {
    let [(left, right)] = padding;
    if left == right {
        return avg_pool1d(input, kernel_size, stride, left, count_include_pad, ceil_mode);
    }
    if count_include_pad {
        return avg_pool1d(input.pad(padding, PadMode::Constant(0.0)),
            kernel_size, stride, 0, true, ceil_mode);
    }
    average_excluding_explicit_padding(input, padding,
        |input| avg_pool1d(input, kernel_size, stride, 0, true, ceil_mode))
}

/// Average pooling with explicit height and width `(before, after)` pairs.
///
/// Half-storage exclusion statistics use FP32; F64 inputs retain F64 statistics.
/// The output retains the input storage, including its native gradient path.
pub fn avg_pool2d_padded<B: Backend>(
    input: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [(usize, usize); 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> Tensor<B, 4> {
    if padding.iter().all(|(start, end)| start == end) {
        return avg_pool2d(input, kernel_size, stride, padding.map(|(start, _)| start),
            count_include_pad, ceil_mode);
    }
    if count_include_pad {
        return avg_pool2d(input.pad(padding, PadMode::Constant(0.0)),
            kernel_size, stride, [0; 2], true, ceil_mode);
    }
    average_excluding_explicit_padding(input, padding,
        |input| avg_pool2d(input, kernel_size, stride, [0; 2], true, ceil_mode))
}

/// Average pooling with depth, height and width `(before, after)` pairs.
///
/// Coverage statistics broadcast across batch/channels and stay on the device.
pub fn avg_pool3d_padded<B: Backend>(
    input: Tensor<B, 5>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [(usize, usize); 3],
    count_include_pad: bool,
    ceil_mode: bool,
) -> Tensor<B, 5> {
    if padding.iter().all(|(start, end)| start == end) {
        return avg_pool3d(input, kernel_size, stride, padding.map(|(start, _)| start),
            count_include_pad, ceil_mode);
    }
    if count_include_pad {
        return avg_pool3d(input.pad(padding, PadMode::Constant(0.0)),
            kernel_size, stride, [0; 3], true, ceil_mode);
    }
    average_excluding_explicit_padding(input, padding,
        |input| avg_pool3d(input, kernel_size, stride, [0; 3], true, ceil_mode))
}

/// Maximum pooling with explicit `(left, right)` padding and native dilation.
pub fn max_pool1d_padded<B: Backend>(
    input: Tensor<B, 3>,
    kernel_size: usize,
    stride: usize,
    padding: [(usize, usize); 1],
    dilation: usize,
    ceil_mode: bool,
) -> Tensor<B, 3> {
    let [(left, right)] = padding;
    let (input, padding) = if left == right {
        (input, left)
    } else {
        (input.pad(padding, PadMode::Constant(f32::NEG_INFINITY)), 0)
    };
    max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)
}

/// Maximum pooling with explicit height and width `(before, after)` pairs.
pub fn max_pool2d_padded<B: Backend>(
    input: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [(usize, usize); 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> Tensor<B, 4> {
    let (input, padding) = if padding.iter().all(|(start, end)| start == end) {
        (input, padding.map(|(start, _)| start))
    } else {
        (input.pad(padding, PadMode::Constant(f32::NEG_INFINITY)), [0; 2])
    };
    max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
}

/// Maximum pooling with explicit depth, height and width padding pairs.
pub fn max_pool3d_padded<B: Backend>(
    input: Tensor<B, 5>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [(usize, usize); 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> Tensor<B, 5> {
    let (input, padding) = if padding.iter().all(|(start, end)| start == end) {
        (input, padding.map(|(start, _)| start))
    } else {
        (input.pad(padding, PadMode::Constant(f32::NEG_INFINITY)), [0; 3])
    };
    max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)
}
