use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::{
    dsl::{calculate_ruda_count_elemwise, prelude::*},
    library::FastDivmod,
    tensor::{RudaTensor, allocation::empty_device_dtype, layout::address_type},
};
use ruda_core::tensor::{DType, Shape, spatial::GridSampleOptions};
use crate::interpolation::bicubic::cubic_coefficient;
use super::base::{PaddingMode, fetch_with_border, reflect_coord};

#[ruda]
fn coordinate<A: Float>(value: A, size: u32, #[comptime] align: bool) -> A {
    let value = select(value != value, A::new(-1.0), value);
    if align { (value + A::new(1.0)) * A::cast_from(size - 1) / A::new(2.0) }
    else { (value + A::new(1.0)) * A::cast_from(size) / A::new(2.0) - A::new(0.5) }
}

#[ruda]
fn bounded<A: Float>(value: A, size: u32, #[comptime] padding: PaddingMode, #[comptime] align: bool) -> A {
    match padding {
        PaddingMode::Zeros => value,
        PaddingMode::Border => clamp(value, A::new(0.0), A::cast_from(size - 1)),
        PaddingMode::Reflection => clamp(reflect_coord::<A>(value, size, align), A::new(0.0), A::cast_from(size - 1)),
    }
}

#[ruda(launch, address_type = "dynamic")]
fn bicubic_kernel<F: Float, A: Float>(
    input: &Tensor<F>, grid: &Tensor<F>, output: &mut Tensor<F>,
    spatial: Sequence<FastDivmod<usize>>,
    #[comptime] align: bool,
    #[comptime] padding: PaddingMode,
    #[define(F)] _dtype: StorageType,
    #[define(A)] _accumulator: StorageType,
) {
    let position = ABSOLUTE_POS;
    if position >= output.shape(0) * output.shape(2) * output.shape(3) { terminate!(); }
    let (remaining, ow) = spatial[2].div_mod(position);
    let (n, oh) = spatial[1].div_mod(remaining);
    let height = input.shape(2) as u32;
    let width = input.shape(3) as u32;
    let grid_base = n * grid.stride(0) + oh * grid.stride(1) + ow * grid.stride(2);
    let x = coordinate::<A>(A::cast_from(grid[grid_base]), width, align);
    let y = coordinate::<A>(A::cast_from(grid[grid_base + grid.stride(3)]), height, align);
    let x0 = x.floor();
    let y0 = y.floor();
    let tx = Vector::<A, Const<1>>::new(x - x0);
    let ty = Vector::<A, Const<1>>::new(y - y0);
    for c in 0..input.shape(1) {
        let input_base = n * input.stride(0) + c * input.stride(1);
        let mut sum = A::new(0.0);
        #[unroll]
        for ky in 0..4usize {
            let yy = bounded::<A>(y0 + A::new(comptime![ky as f32 - 1.0]), height, padding, align);
            let wy = cubic_coefficient::<A, Const<1>>(ty, ky)[0];
            let mut row = A::new(0.0);
            #[unroll]
            for kx in 0..4usize {
                let xx = bounded::<A>(x0 + A::new(comptime![kx as f32 - 1.0]), width, padding, align);
                let invalid = xx != xx || yy != yy || xx < A::new(0.0) || yy < A::new(0.0)
                    || xx >= A::cast_from(width) || yy >= A::cast_from(height);
                let wx = cubic_coefficient::<A, Const<1>>(tx, kx)[0];
                let mut value = A::new(0.0);
                if !invalid {
                    value = A::cast_from(fetch_with_border(input, input_base, input.stride(2), input.stride(3),
                        i32::cast_from(yy), i32::cast_from(xx), height as i32, width as i32));
                }
                row += value * wx;
            }
            sum += row * wy;
        }
        let output_index = n * output.stride(0) + c * output.stride(1)
            + oh * output.stride(2) + ow * output.stride(3);
        output[output_index] = F::cast_from(sum);
    }
}

pub(super) fn launch<R: Runtime>(input: RudaTensor<R>, grid: RudaTensor<R>, options: GridSampleOptions) -> RudaTensor<R> {
    let [batch, channels, height, width] = input.meta.shape().dims();
    let [grid_batch, out_h, out_w, coordinates] = grid.meta.shape().dims();
    assert_eq!(batch, grid_batch);
    assert_eq!(coordinates, 2);
    assert!(height > 0 && width > 0);
    assert!(height <= i32::MAX as usize && width <= i32::MAX as usize);
    assert_eq!(input.dtype, grid.dtype);
    let accumulator = match input.dtype {
        DType::F64 => DType::F64,
        DType::F32 | DType::F16 | DType::BF16 => DType::F32,
        dtype => panic!("grid_sample_2d bicubic: unsupported dtype {dtype:?}"),
    };
    let output = empty_device_dtype(input.client.clone(), input.device.clone(),
        Shape::new([batch, channels, out_h, out_w]), input.dtype);
    if output.meta.shape().num_elements() == 0 { return output; }
    let count = batch * out_h * out_w;
    let mut spatial = SequenceArg::new();
    for size in [batch, out_h, out_w] { spatial.push(size); }
    let ruda_dim = RudaDim::new(input.client.properties(), count);
    let ruda_count = calculate_ruda_count_elemwise(&input.client, count, ruda_dim);
    let dtype = input.dtype;
    bicubic_kernel::launch(&output.client, ruda_count, ruda_dim, address_type!(input, grid, output),
        input.into_tensor_arg(), grid.into_tensor_arg(), output.clone().into_tensor_arg(),
        spatial, options.align_corners, options.padding_mode.into(), dtype.into(), accumulator.into());
    output
}
