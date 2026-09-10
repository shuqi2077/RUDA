use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::{
    dsl::{calculate_ruda_count_elemwise, prelude::*},
    library::FastDivmod,
    tensor::{RudaTensor, allocation::empty_device_dtype, layout::address_type},
};
use ruda_core::tensor::{Shape, spatial::GridSampleOptions};
use super::base::{PaddingMode, fetch_with_border, reflect_coord};

#[ruda]
fn rounded<F: Float>(value: F) -> F {
    let value = select(value != value, F::new(0.0), value);
    let absolute = value.abs();
    let floor = absolute.floor();
    let magnitude = select(absolute - floor >= F::new(0.5), floor + F::new(1.0), floor);
    select(value < F::new(0.0), -magnitude, magnitude)
}

#[ruda(launch, address_type = "dynamic")]
fn nearest_kernel<F: Float>(
    input: &Tensor<F>,
    grid: &Tensor<F>,
    output: &mut Tensor<F>,
    spatial: Sequence<FastDivmod<usize>>,
    #[comptime] align: bool,
    #[comptime] padding: PaddingMode,
    #[define(F)] _dtype: StorageType,
) {
    let position = ABSOLUTE_POS;
    if position >= output.shape(0) * output.shape(2) * output.shape(3) { terminate!(); }
    let (remaining, ow) = spatial[2].div_mod(position);
    let (n, oh) = spatial[1].div_mod(remaining);
    let height = input.shape(2) as u32;
    let width = input.shape(3) as u32;
    let base = n * grid.stride(0) + oh * grid.stride(1) + ow * grid.stride(2);
    let gx = grid[base];
    let gy = grid[base + grid.stride(3)];
    let (px, py) = if align {
        ((gx + F::new(1.0)) * F::cast_from(width - 1) / F::new(2.0),
         (gy + F::new(1.0)) * F::cast_from(height - 1) / F::new(2.0))
    } else {
        ((gx + F::new(1.0)) * F::cast_from(width) / F::new(2.0) - F::new(0.5),
         (gy + F::new(1.0)) * F::cast_from(height) / F::new(2.0) - F::new(0.5))
    };
    let max_x = F::cast_from(width - 1);
    let max_y = F::cast_from(height - 1);
    let (px, py) = match padding {
        PaddingMode::Border => {
            let nonfinite = px != px || py != py
                || px.abs() == F::new(f32::INFINITY) || py.abs() == F::new(f32::INFINITY);
            (clamp(select(nonfinite, max_x / F::new(2.0), px), F::new(0.0), max_x),
             clamp(select(nonfinite, max_y / F::new(2.0), py), F::new(0.0), max_y))
        }
        PaddingMode::Reflection => (reflect_coord::<F>(px, width, align), reflect_coord::<F>(py, height, align)),
        PaddingMode::Zeros => (px, py),
    };
    let rx = rounded::<F>(px);
    let ry = rounded::<F>(py);
    let outside = rx < F::new(0.0) || rx > max_x || ry < F::new(0.0) || ry > max_y;
    let ix = i32::cast_from(clamp(rx, F::new(0.0), max_x));
    let iy = i32::cast_from(clamp(ry, F::new(0.0), max_y));
    for channel in 0..input.shape(1) {
        let output_index = n * output.stride(0) + channel * output.stride(1)
            + oh * output.stride(2) + ow * output.stride(3);
        if comptime!(padding == PaddingMode::Zeros) && outside {
            output[output_index] = F::new(0.0);
        } else {
            let input_base = n * input.stride(0) + channel * input.stride(1);
            output[output_index] = fetch_with_border(input, input_base, input.stride(2), input.stride(3),
                iy, ix, height as i32, width as i32);
        }
    }
}

pub(super) fn launch<R: Runtime>(input: RudaTensor<R>, grid: RudaTensor<R>, options: GridSampleOptions) -> RudaTensor<R> {
    let [batch, channels, height, width] = input.meta.shape().dims();
    let [grid_batch, out_h, out_w, coordinates] = grid.meta.shape().dims();
    assert_eq!(grid_batch, batch);
    assert_eq!(coordinates, 2);
    assert!(height > 0 && width > 0);
    assert!(height <= i32::MAX as usize && width <= i32::MAX as usize);
    assert_eq!(input.dtype, grid.dtype);
    let output = empty_device_dtype(input.client.clone(), input.device.clone(),
        Shape::new([batch, channels, out_h, out_w]), input.dtype);
    if output.meta.shape().num_elements() == 0 { return output; }
    let count = batch * out_h * out_w;
    let mut spatial = SequenceArg::new();
    for size in [batch, out_h, out_w] { spatial.push(size); }
    let ruda_dim = RudaDim::new(input.client.properties(), count);
    let ruda_count = calculate_ruda_count_elemwise(&input.client, count, ruda_dim);
    let dtype = input.dtype;
    nearest_kernel::launch(&output.client, ruda_count, ruda_dim, address_type!(input, grid, output),
        input.into_tensor_arg(), grid.into_tensor_arg(), output.clone().into_tensor_arg(),
        spatial, options.align_corners, options.padding_mode.into(), dtype.into());
    output
}
