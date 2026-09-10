use ruda_kernel::dsl as kernel_dsl;
use super::pool2d::{
    Pool2dDirectArgsLaunch, Pool2dDirectStrategy, Pool2dDirectStrategyFamily, pool2d_direct,
};
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::contiguous::into_contiguous_aligned;
use crate::pooling::pool2d::Position;
use crate::pooling::pool2d::view4d;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::layout::max_vector_size;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::permutation::permute_nchw_to_nhwc;
use ruda_kernel::tensor::permutation::permute_nhwc_to_nchw;
use ruda_kernel::tensor::RudaTensor;
use ruda_core::tensor::Shape;
use ruda_core::tensor::spatial::calculate_pool_output_size;
use ruda_kernel::dsl::RudaDim;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::num_traits::Zero;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::View;

struct AvgPoolStrategy;

impl Pool2dDirectStrategyFamily for AvgPoolStrategy {
    type Indices<I: Int, N: Size> = ();
    type Config = AvgPoolStrategyConfig;
    type Pool2d<T: Numeric, I: Int, N: Size> = Self;
}

#[derive(RudaType, Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct AvgPoolStrategyConfig {
    count_include_pad: bool,
    /// Total padded height (input_height + 2 * padding_0)
    padded_h: u32,
    /// Total padded width (input_width + 2 * padding_1)
    padded_w: u32,
}

#[ruda]
impl<T: Numeric, I: Int, N: Size> Pool2dDirectStrategy<T, I, N> for AvgPoolStrategy {
    type Accumulator = (Vector<T, N>, u32);
    type Config = AvgPoolStrategyConfig;
    type Indices = ();

    fn initialize(#[comptime] _config: &Self::Config) -> Self::Accumulator {
        let sum = Vector::zero();
        // Count will be set dynamically: either by accumulate (count_include_pad=false)
        // or by set_padded_count (count_include_pad=true)
        let count = 0u32;

        (sum, count)
    }

    fn accumulate(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        _index: usize,
        result: Vector<T, N>,
    ) {
        let (sum, count) = accumulator;

        // Only count valid positions when count_include_pad=false
        if comptime![!config.count_include_pad] {
            *count += 1;
        }

        *sum += result;
    }

    fn count_position(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        ih: u32,
        iw: u32,
    ) {
        // When count_include_pad=true, count positions within padded bounds
        // (excludes ceil_mode extensions beyond the padded input)
        if comptime![config.count_include_pad] && ih < config.padded_h && iw < config.padded_w {
            let (_sum, count) = accumulator;
            *count += 1;
        }
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: Position,
        output: &mut View<Vector<T, N>, Position, ReadWrite>,
        _output_indices: &mut (),
        accumulator: Self::Accumulator,
    ) {
        let (sum, count) = accumulator;
        output[position] = sum / Vector::cast_from(count);
    }
}

pub fn avg_pool2d<R: Runtime>(
    x: RudaTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> RudaTensor<R> {
    let [batch_size, channels, in_h, in_w] = x.meta.shape().dims();
    let dilation = 1;

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation,
        in_h,
        ceil_mode,
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation,
        in_w,
        ceil_mode,
    );

    // Padded dimensions (for count_include_pad with ceil_mode)
    let padded_0 = in_h + 2 * padding[0];
    let padded_1 = in_w + 2 * padding[1];

    let x = into_contiguous_aligned(permute_nchw_to_nhwc(x));
    let vector_size = max_vector_size(&x);

    let shape_out = Shape::new([batch_size, size_0, size_1, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), shape_out, x.dtype);

    let working_units = output.meta.num_elements() / vector_size as usize;
    let ruda_dim = RudaDim::new(x.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&x.client, working_units, ruda_dim);

    pool2d_direct::launch::<AvgPoolStrategy, R>(
        &output.client,
        ruda_count,
        ruda_dim,
        address_type!(x, output),
        vector_size,
        x.into_tensor_arg(),
        view4d(output.clone(), vector_size),
        (),
        shape_divmod(&output),
        working_units,
        Pool2dDirectArgsLaunch::new(
            stride[0] as u32,
            stride[1] as u32,
            dilation as u32,
            dilation as u32,
            padding[0] as u32,
            padding[1] as u32,
        ),
        (kernel_size[0] as u32, kernel_size[1] as u32),
        AvgPoolStrategyConfig {
            count_include_pad,
            padded_h: padded_0 as u32,
            padded_w: padded_1 as u32,
        },
        [output.dtype.into(), ruda_core::tensor::DType::I32.into()],
    );

    permute_nhwc_to_nchw(output)
}
