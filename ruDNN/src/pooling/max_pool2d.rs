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
use ruda_core::tensor::DType;
use ruda_core::tensor::Shape;
use ruda_core::tensor::spatial::calculate_pool_output_size;
use ruda_kernel::dsl::RudaDim;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::num_traits::Zero;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::View;

struct MaxPoolStrategy;
struct MaxPoolWithIndicesStrategy;

pub(super) fn validate_index_dtype(dtype: DType, height: usize, width: usize) {
    let capacity = match dtype {
        DType::I8 => i8::MAX as u128 + 1,
        DType::I16 => i16::MAX as u128 + 1,
        DType::I32 => i32::MAX as u128 + 1,
        DType::I64 => i64::MAX as u128 + 1,
        DType::U8 => u8::MAX as u128,
        DType::U16 => u16::MAX as u128,
        DType::U32 => u32::MAX as u128,
        DType::U64 => u64::MAX as u128,
        _ => panic!("Max pooling indices require an integer dtype, got {dtype:?}"),
    };
    let elements = height.checked_mul(width).expect("Max pooling input plane size overflow");
    assert!(elements as u128 <= capacity, "Max pooling input plane does not fit {dtype:?} indices with an empty-window sentinel");
}

impl Pool2dDirectStrategyFamily for MaxPoolStrategy {
    type Indices<I: Int, N: Size> = ();
    type Config = bool;
    type Pool2d<T: Numeric, I: Int, N: Size> = Self;
}

impl Pool2dDirectStrategyFamily for MaxPoolWithIndicesStrategy {
    type Indices<I: Int, N: Size> = View<Vector<I, N>, Position, ReadWrite>;
    type Config = bool;
    type Pool2d<T: Numeric, I: Int, N: Size> = Self;
}

#[ruda]
impl<T: Numeric, I: Int, N: Size> Pool2dDirectStrategy<T, I, N> for MaxPoolStrategy {
    type Accumulator = (Vector<T, N>, bool);
    type Config = bool;
    type Indices = ();

    fn initialize(#[comptime] config: &Self::Config) -> Self::Accumulator {
        let value = if comptime![config.clone()] {
            Vector::cast_from(f32::NEG_INFINITY)
        } else {
            Vector::new(T::min_value())
        };
        (value, false)
    }

    fn accumulate(
        #[comptime] _config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        _index: VectorSize,
        result: Vector<T, N>,
    ) {
        if !accumulator.1 {
            accumulator.0 = result;
            accumulator.1 = true;
        } else {
            accumulator.0 = select_many(result.greater_than(accumulator.0), result, accumulator.0);
        }
    }

    fn count_position(
        #[comptime] _config: &Self::Config,
        _accumulator: &mut Self::Accumulator,
        _ih: u32,
        _iw: u32,
    ) {
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: Position,
        output: &mut View<Vector<T, N>, Position, ReadWrite>,
        _output_indices: &mut (),
        accumulator: Self::Accumulator,
    ) {
        output[position] = accumulator.0;
    }
}

#[ruda]
impl<T: Numeric, I: Int, N: Size> Pool2dDirectStrategy<T, I, N> for MaxPoolWithIndicesStrategy {
    type Accumulator = (Vector<T, N>, Vector<I, N>, bool);
    type Config = bool;
    type Indices = View<Vector<I, N>, Position, ReadWrite>;

    fn initialize(#[comptime] config: &Self::Config) -> Self::Accumulator {
        let val = if comptime![config.clone()] {
            Vector::cast_from(f32::NEG_INFINITY)
        } else {
            Vector::new(T::min_value())
        };
        let idx = Vector::new(I::new(-1));
        (val, idx, false)
    }

    fn accumulate(
        #[comptime] _config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        index: usize,
        result: Vector<T, N>,
    ) {
        let indices = Vector::cast_from(index);
        if !accumulator.2 {
            accumulator.0 = result;
            accumulator.1 = indices;
            accumulator.2 = true;
        } else {
            let replace = result.greater_than(accumulator.0);
            accumulator.1 = select_many(replace, indices, accumulator.1);
            accumulator.0 = select_many(replace, result, accumulator.0);
        }
    }

    fn count_position(
        #[comptime] _config: &Self::Config,
        _accumulator: &mut Self::Accumulator,
        _ih: u32,
        _iw: u32,
    ) {
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: Position,
        output: &mut View<Vector<T, N>, Position, ReadWrite>,
        output_indices: &mut View<Vector<I, N>, Position, ReadWrite>,
        accumulator: Self::Accumulator,
    ) {
        output[position] = accumulator.0;
        output_indices[position] = accumulator.1;
    }
}

pub fn max_pool2d<R: Runtime>(
    x: RudaTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> RudaTensor<R> {
    let [batch_size, channels, height, width] = x.meta.shape().dims();

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        height,
        ceil_mode,
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        width,
        ceil_mode,
    );

    let x = into_contiguous_aligned(permute_nchw_to_nhwc(x));

    let vector_size = max_vector_size(&x);

    let shape_out = Shape::new([batch_size, size_0, size_1, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), shape_out, x.dtype);

    let working_units = output.meta.num_elements() / vector_size as usize;
    let ruda_dim = RudaDim::new(x.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&x.client, working_units, ruda_dim);

    pool2d_direct::launch::<MaxPoolStrategy, R>(
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
            dilation[0] as u32,
            dilation[1] as u32,
            padding[0] as u32,
            padding[1] as u32,
        ),
        (kernel_size[0] as u32, kernel_size[1] as u32),
        matches!(output.dtype, DType::F16 | DType::BF16 | DType::F32 | DType::Flex32 | DType::F64),
        [output.dtype.into(), DType::I32.into()],
    );

    permute_nhwc_to_nchw(output)
}

pub fn max_pool2d_with_indices<R: Runtime>(
    x: RudaTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    dtype_indices: DType,
) -> (RudaTensor<R>, RudaTensor<R>) {
    let [batch_size, channels, size_0, size_1] = x.meta.shape().dims();
    validate_index_dtype(dtype_indices, size_0, size_1);

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        size_0,
        ceil_mode,
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        size_1,
        ceil_mode,
    );

    let x = into_contiguous_aligned(permute_nchw_to_nhwc(x));
    let vector_size = max_vector_size(&x);

    let shape_out = Shape::new([batch_size, size_0, size_1, channels]);
    let output = empty_device_dtype(
        x.client.clone(),
        x.device.clone(),
        shape_out.clone(),
        x.dtype,
    );
    let indices = empty_device_dtype(x.client.clone(), x.device.clone(), shape_out, dtype_indices);

    let working_units = output.meta.num_elements() / vector_size as usize;
    let ruda_dim = RudaDim::new(x.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&x.client, working_units, ruda_dim);

    pool2d_direct::launch::<MaxPoolWithIndicesStrategy, R>(
        &output.client,
        ruda_count,
        ruda_dim,
        address_type!(x, output, indices),
        vector_size,
        x.into_tensor_arg(),
        view4d(output.clone(), vector_size),
        view4d(indices.clone(), vector_size),
        shape_divmod(&output),
        working_units,
        Pool2dDirectArgsLaunch::new(
            stride[0] as u32,
            stride[1] as u32,
            dilation[0] as u32,
            dilation[1] as u32,
            padding[0] as u32,
            padding[1] as u32,
        ),
        (kernel_size[0] as u32, kernel_size[1] as u32),
        matches!(output.dtype, DType::F16 | DType::BF16 | DType::F32 | DType::Flex32 | DType::F64),
        [output.dtype.into(), dtype_indices.into()],
    );

    let output = permute_nhwc_to_nchw(output);
    let indices = permute_nhwc_to_nchw(indices);
    (output, indices)
}
