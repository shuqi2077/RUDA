use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::dsl::{calculate_ruda_count_elemwise, RudaDim};
use ruda_kernel::library::tensor::View;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype,
    contiguous::into_contiguous_aligned, layout::{address_type, max_vector_size, shape_divmod},
    permutation::{permute_nchw_to_nhwc, permute_nhwc_to_nchw}};
use ruda_core::tensor::{DType, Shape};
use super::pool2d::{Position, Pool2dDirectArgsLaunch, Pool2dDirectStrategy,
    Pool2dDirectStrategyFamily, pool2d_direct, view4d};

struct AtenMaxPool;

#[derive(RudaType, Debug, PartialEq, Eq, Hash, Clone, Copy)]
struct AtenMaxPoolConfig {
    stride_h: u32,
    stride_w: u32,
    padding_h: u32,
    padding_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    width: usize,
    channels_last: bool,
}

impl Pool2dDirectStrategyFamily for AtenMaxPool {
    type Indices<I: Int, N: Size> = View<Vector<I, N>, Position, ReadWrite>;
    type Config = AtenMaxPoolConfig;
    type Pool2d<T: Numeric, I: Int, N: Size> = Self;
}

#[ruda]
impl<T: Numeric, I: Int, N: Size> Pool2dDirectStrategy<T, I, N> for AtenMaxPool {
    type Accumulator = (Vector<T, N>, Vector<I, N>);
    type Config = AtenMaxPoolConfig;
    type Indices = View<Vector<I, N>, Position, ReadWrite>;

    fn initialize(#[comptime] _config: &Self::Config) -> Self::Accumulator {
        (Vector::cast_from(f32::NEG_INFINITY), Vector::new(I::new(-1)))
    }

    fn accumulate(
        #[comptime] _config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        index: usize,
        result: Vector<T, N>,
    ) {
        let replace = result.greater_than(accumulator.0).or(result.not_equal(result));
        accumulator.0 = select_many(replace, result, accumulator.0);
        accumulator.1 = select_many(replace, Vector::cast_from(index), accumulator.1);
    }

    fn count_position(
        #[comptime] _config: &Self::Config,
        _accumulator: &mut Self::Accumulator,
        _ih: u32,
        _iw: u32,
    ) {}

    fn store(
        #[comptime] config: &Self::Config,
        position: Position,
        output: &mut View<Vector<T, N>, Position, ReadWrite>,
        output_indices: &mut Self::Indices,
        accumulator: Self::Accumulator,
    ) {
        let mut initial_index = 0i64;
        if comptime![!config.channels_last] {
            let mut h = position.1 as i64 * config.stride_h as i64 - config.padding_h as i64;
            let mut w = position.2 as i64 * config.stride_w as i64 - config.padding_w as i64;
            while h < 0 {
                h += config.dilation_h as i64;
            }
            while w < 0 {
                w += config.dilation_w as i64;
            }
            initial_index = h * config.width as i64 + w;
        }
        output[position] = accumulator.0;
        output_indices[position] = select_many(accumulator.1.equal(Vector::new(I::new(-1))),
            Vector::cast_from(initial_index), accumulator.1);
    }
}

pub fn max_pool2d_with_indices_aten<R: Runtime>(
    x: RudaTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    channels_last: bool,
) -> (RudaTensor<R>, RudaTensor<R>) {
    assert!(matches!(x.dtype, DType::F32 | DType::F16 | DType::BF16));
    let [batch, channels, height, width] = x.meta.shape().dims();
    super::max_pool2d::validate_index_dtype(DType::I64, height, width);
    let output_size = |axis: usize, input: usize| {
        assert!(kernel_size[axis] > 0 && stride[axis] > 0 && dilation[axis] > 0);
        let step = stride[axis] as i128;
        let extent = dilation[axis] as i128 * (kernel_size[axis] as i128 - 1) + 1;
        let numerator = input as i128 + 2 * padding[axis] as i128 - extent;
        let mut size = (numerator + if ceil_mode { step - 1 } else { 0 }).div_euclid(step) + 1;
        if ceil_mode && (size - 1) * step >= input as i128 + padding[axis] as i128 {
            size -= 1;
        }
        assert!(size > 0, "max pooling output must be positive");
        usize::try_from(size).expect("max pooling output size overflow")
    };
    let shape = Shape::new([batch, output_size(0, height), output_size(1, width), channels]);
    let x = into_contiguous_aligned(permute_nchw_to_nhwc(x));
    let vector_size = max_vector_size(&x);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), shape.clone(), x.dtype);
    let indices = empty_device_dtype(x.client.clone(), x.device.clone(), shape, DType::I64);
    let working_units = output.meta.num_elements() / vector_size as usize;
    if working_units != 0 {
        let dim = RudaDim::new(x.client.properties(), working_units);
        let count = calculate_ruda_count_elemwise(&x.client, working_units, dim);
        pool2d_direct::launch::<AtenMaxPool, R>(
            &output.client, count, dim, address_type!(x, output, indices), vector_size,
            x.into_tensor_arg(), view4d(output.clone(), vector_size), view4d(indices.clone(), vector_size),
            shape_divmod(&output), working_units,
            Pool2dDirectArgsLaunch::new(stride[0] as u32, stride[1] as u32,
                dilation[0] as u32, dilation[1] as u32, padding[0] as u32, padding[1] as u32),
            (kernel_size[0] as u32, kernel_size[1] as u32),
            AtenMaxPoolConfig { stride_h: stride[0] as u32, stride_w: stride[1] as u32,
                padding_h: padding[0] as u32, padding_w: padding[1] as u32,
                dilation_h: dilation[0] as u32, dilation_w: dilation[1] as u32, width, channels_last },
            [output.dtype.into(), DType::I64.into()],
        );
    }
    (permute_nhwc_to_nchw(output), permute_nhwc_to_nchw(indices))
}
