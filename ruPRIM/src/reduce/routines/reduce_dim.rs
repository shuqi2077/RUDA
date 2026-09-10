use ruda_kernel::dsl as kernel_dsl;
use crate::reduce::{ReducePrecision, components::args::NumericVector, routines::GlobalReduceBlueprint};
use ruda_kernel::dsl::prelude::ReadWrite;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::r#virtual::VirtualTensor;

#[ruda]
pub trait ReduceDimRoutine {
    type Config;

    fn execute<P: ReducePrecision, Out: NumericVector>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        axis_reduce: u32,
        reduce_index: u32,
        #[comptime] config: Self::Config,
    );

    fn create_config(#[comptime] blueprint: GlobalReduceBlueprint) -> comptime_type!(Self::Config);
}
