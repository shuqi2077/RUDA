use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::max_vector_size;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::RudaTensor;
use ruda_core::tensor::TensorMetadata;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::layout::linear::LinearView;

pub trait FloatUnaryOpFamily: 'static + Send + Sync {
    type Options: LaunchArg;
    type Unary<F: Float, N: Size>: FloatUnaryOp<F, N, Options = Self::Options>;
}

#[ruda]
pub trait FloatUnaryOp<F: Float, N: Size>: 'static + Send + Sync {
    type Options: LaunchArg;

    fn execute(input: Vector<F, N>, options: &Self::Options) -> Vector<F, N>;
}

#[ruda(launch_unchecked, address_type = "dynamic")]
pub fn unary_float<F: Float, N: Size, O: FloatUnaryOpFamily>(
    input: &LinearView<Vector<F, N>>,
    output: &mut LinearView<Vector<F, N>, ReadWrite>,
    options: &O::Options,
    #[define(F)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = O::Unary::<F, N>::execute(input[ABSOLUTE_POS], options);
}

pub fn launch_unary_float<R, O, Args>(tensor: RudaTensor<R>, args: Args) -> RudaTensor<R>
where
    // Magic fix for lifetime, the closure is supposed to capture everything required to create the
    // argument.
    for<'a> Args: FnOnce(&'a ()) -> RuntimeArg<O::Options, R>,
    R: Runtime,
    O: FloatUnaryOpFamily,
{
    let vector_size = max_vector_size(&tensor);

    let client = tensor.client.clone();
    let num_elems = tensor.meta.num_elements();

    let working_units = num_elems / vector_size as usize;
    let ruda_dim = RudaDim::new(tensor.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&tensor.client, working_units, ruda_dim);
    let dtype = tensor.dtype;

    unsafe {
        if tensor.can_mut() && tensor.is_nonoverlapping() {
            unary_float::launch_unchecked::<O, R>(
                &client,
                ruda_count,
                ruda_dim,
                address_type!(tensor),
                vector_size,
                tensor.clone().into_linear_view(),
                tensor.as_linear_view_alias(0),
                args(&()),
                dtype.into(),
            );

            tensor
        } else {
            let output = empty_device_dtype(
                tensor.client.clone(),
                tensor.device.clone(),
                tensor.shape(),
                tensor.dtype,
            );

            unary_float::launch_unchecked::<O, R>(
                &client,
                ruda_count,
                ruda_dim,
                address_type!(tensor, output),
                vector_size,
                tensor.into_linear_view(),
                output.clone().into_linear_view(),
                args(&()),
                dtype.into(),
            );

            output
        }
    }
}

/// Use comptime enum to implement all unary operations that don't have any input argument in the
/// kernel definition.
pub mod unary_basic {
    use ruda_kernel::dsl::num_traits::One;
    use ruda_kernel::dsl::num_traits::Zero;

    use super::*;

    pub fn launch<R, Args>(tensor: RudaTensor<R>, args: Args) -> RudaTensor<R>
    where
        R: Runtime,
        for<'a> Args: FnOnce(&'a ()) -> BasicFloatUnaryKind,
    {
        launch_unary_float::<R, BasicFloatUnary, _>(tensor, |input| {
            BasicFloatUnaryOptionsLaunch::new(args(input))
        })
    }

    #[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    pub enum BasicFloatUnaryKind {
        Exp,
        Log,
        Log1p,
        Sqrt,
        Abs,
        Sign,
        ArcCos,
        ArcCosh,
        ArcSin,
        ArcSinh,
        ArcTan,
        ArcTanh,
        Cos,
        Cosh,
        Sin,
        Sinh,
        Tan,
        Tanh,
        Round,
        Floor,
        Ceil,
        Trunc,
        Erf,
        Recip,
        InverseSqrt,
    }

    #[derive(RudaLaunch, RudaType)]
    struct BasicFloatUnaryOptions {
        #[ruda(comptime)]
        kind: BasicFloatUnaryKind,
    }
    struct BasicFloatUnary;

    #[ruda]
    impl<F: Float, N: Size> FloatUnaryOp<F, N> for BasicFloatUnary {
        type Options = BasicFloatUnaryOptions;

        fn execute(input: Vector<F, N>, options: &Self::Options) -> Vector<F, N> {
            match comptime![options.kind] {
                BasicFloatUnaryKind::Exp => Vector::exp(input),
                BasicFloatUnaryKind::Log => Vector::ln(input),
                BasicFloatUnaryKind::Log1p => Vector::log1p(input),
                BasicFloatUnaryKind::Sqrt => Vector::sqrt(input),
                BasicFloatUnaryKind::Abs => Vector::abs(input),
                BasicFloatUnaryKind::Sign => {
                    let zero = Vector::zero();
                    let one = Vector::one();
                    let minus_one = Vector::new(F::new(-1.0));

                    let is_positive = input.greater_than(zero);
                    let is_negative = input.less_than(zero);
                    let sign = select_many(is_negative, minus_one, zero);

                    select_many(is_positive, one, sign)
                }
                BasicFloatUnaryKind::Cos => Vector::cos(input),
                BasicFloatUnaryKind::Sin => Vector::sin(input),
                BasicFloatUnaryKind::Tan => Vector::tan(input),
                BasicFloatUnaryKind::Cosh => Vector::cosh(input),
                BasicFloatUnaryKind::Sinh => Vector::sinh(input),
                BasicFloatUnaryKind::Tanh => Vector::tanh(input),
                BasicFloatUnaryKind::Round => Vector::round(input),
                BasicFloatUnaryKind::Floor => Vector::floor(input),
                BasicFloatUnaryKind::Ceil => Vector::ceil(input),
                BasicFloatUnaryKind::Trunc => Vector::trunc(input),
                BasicFloatUnaryKind::Erf => Vector::erf(input),
                BasicFloatUnaryKind::Recip => Vector::recip(input),
                BasicFloatUnaryKind::InverseSqrt => Vector::inverse_sqrt(input),
                BasicFloatUnaryKind::ArcCos => Vector::acos(input),
                BasicFloatUnaryKind::ArcCosh => Vector::acosh(input),
                BasicFloatUnaryKind::ArcSin => Vector::asin(input),
                BasicFloatUnaryKind::ArcSinh => Vector::asinh(input),
                BasicFloatUnaryKind::ArcTan => Vector::atan(input),
                BasicFloatUnaryKind::ArcTanh => Vector::atanh(input),
            }
        }
    }

    impl FloatUnaryOpFamily for BasicFloatUnary {
        type Options = BasicFloatUnaryOptions;
        type Unary<F: Float, N: Size> = Self;
    }
}
