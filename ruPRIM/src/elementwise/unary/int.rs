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

pub trait IntUnaryOpFamily: 'static + Send + Sync {
    type Options: LaunchArg;
    type Unary<I: Int, N: Size>: IntUnaryOp<I, N, Options = Self::Options>;
}

#[ruda]
pub trait IntUnaryOp<I: Scalar, N: Size>: 'static + Send + Sync {
    type Options: LaunchArg;

    fn execute(input: Vector<I, N>, options: &Self::Options) -> Vector<I, N>;
}

#[ruda(launch_unchecked, address_type = "dynamic")]
pub fn unary_int<I: Int, N: Size, O: IntUnaryOpFamily>(
    input: &LinearView<Vector<I, N>>,
    output: &mut LinearView<Vector<I, N>, ReadWrite>,
    options: &O::Options,
    #[define(I)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = O::Unary::<I, N>::execute(input[ABSOLUTE_POS], options);
}

pub fn launch_unary_int<R, O, Args>(tensor: RudaTensor<R>, args: Args) -> RudaTensor<R>
where
    for<'a> Args: FnOnce(&'a ()) -> RuntimeArg<O::Options, R>,
    R: Runtime,
    O: IntUnaryOpFamily,
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
            unary_int::launch_unchecked::<O, R>(
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

            unary_int::launch_unchecked::<O, R>(
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

pub mod unary_basic_int {

    use ruda_kernel::dsl::num_traits::One;
    use ruda_kernel::dsl::num_traits::Zero;

    use super::*;

    pub fn launch<R, Args>(tensor: RudaTensor<R>, args: Args) -> RudaTensor<R>
    where
        R: Runtime,
        for<'a> Args: FnOnce(&'a ()) -> BasicIntUnaryKind,
    {
        launch_unary_int::<R, BasicIntUnary, _>(tensor, |input| {
            BasicIntUnaryOptionsLaunch::new(args(input))
        })
    }

    #[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    pub enum BasicIntUnaryKind {
        BitwiseNot,
        Sign,
    }

    #[derive(RudaLaunch, RudaType)]
    struct BasicIntUnaryOptions {
        #[ruda(comptime)]
        kind: BasicIntUnaryKind,
    }
    struct BasicIntUnary;

    #[ruda]
    impl<I: Int, N: Size> IntUnaryOp<I, N> for BasicIntUnary {
        type Options = BasicIntUnaryOptions;

        fn execute(input: Vector<I, N>, options: &Self::Options) -> Vector<I, N> {
            match comptime![options.kind] {
                BasicIntUnaryKind::BitwiseNot => !input,
                BasicIntUnaryKind::Sign => {
                    let zero = Vector::zero();
                    let one = Vector::one();
                    let minus_one = Vector::new(I::new(-1));

                    let is_positive = input.greater_than(zero);
                    let is_negative = input.less_than(zero);
                    let sign = select_many(is_negative, minus_one, zero);

                    select_many(is_positive, one, sign)
                }
            }
        }
    }

    impl IntUnaryOpFamily for BasicIntUnary {
        type Options = BasicIntUnaryOptions;
        type Unary<I: Int, N: Size> = Self;
    }
}
