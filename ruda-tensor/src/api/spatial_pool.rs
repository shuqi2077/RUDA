use super::{BasicOps, DType, Int, Tensor, backend::Backend};
use super::module::{
    adaptive_avg_pool1d, adaptive_avg_pool2d, avg_pool1d, avg_pool2d,
    max_pool1d, max_pool2d, max_pool1d_with_indices, max_pool2d_with_indices,
};

fn volume_planes<B: Backend, K: BasicOps<B>>(input: Tensor<B, 5, K>) -> Tensor<B, 4, K> {
    let [batch, channels, depth, height, width] = input.dims();
    let planes = batch.checked_mul(depth).expect("pooling plane count overflow");
    input.permute([0, 2, 1, 3, 4]).reshape([planes, channels, height, width])
}

fn plane_depth_lines<B: Backend, K: BasicOps<B>>(
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

fn depth_lines_volume<B: Backend, K: BasicOps<B>>(
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
/// Indices flatten each input volume as `depth * height * width + height * width
/// + width`, independently for every batch/channel. Tie selection follows the
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
    let spatial_indices = plane_depth_lines(plane_indices.cast(DType::I64), batch, depth)
        .gather(2, depth_indices.clone().clamp(0, depth as i64 - 1));
    let invalid = depth_indices.clone().lower_elem(0)
        .bool_or(depth_indices.clone().greater_equal_elem(depth as i64))
        .bool_or(spatial_indices.clone().lower_elem(0))
        .bool_or(spatial_indices.clone().greater_equal_elem(area as i64));
    let indices = (depth_indices.mul_scalar(area as i64) + spatial_indices).mask_fill(invalid, -1);
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
