use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::unary_numeric::NumericUnaryOp;
use ruda_kernel::tensor::unary_numeric::NumericUnaryOpFamily;
use ruda_kernel::tensor::unary_numeric::launch_unary_numeric;
use ruda_kernel::tensor::RudaTensor;

#[derive(RudaLaunch, RudaType)]
struct Options {
    min_value: InputScalar,
    max_value: InputScalar,
}

pub fn clamp<R: Runtime>(
    input: RudaTensor<R>,
    min_value: InputScalar,
    max_value: InputScalar,
) -> RudaTensor<R> {
    struct ClampOp;

    #[ruda]
    impl<T: Numeric, N: Size> NumericUnaryOp<T, N> for ClampOp {
        type Options = Options;

        fn execute(input: Vector<T, N>, options: &Self::Options) -> Vector<T, N> {
            ruda_kernel::dsl::prelude::clamp(
                input,
                Vector::new(options.min_value.get::<T>()),
                Vector::new(options.max_value.get::<T>()),
            )
        }
    }

    impl NumericUnaryOpFamily for ClampOp {
        type Options = Options;
        type Unary<T: Numeric, N: Size> = Self;
    }

    launch_unary_numeric::<R, ClampOp, _>(input, |_| OptionsLaunch::new(min_value, max_value))
}
