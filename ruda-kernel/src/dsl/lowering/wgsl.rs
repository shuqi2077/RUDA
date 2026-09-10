use alloc::{boxed::Box, string::String};
use ruda_core::{ir::{Processor, Scope, Variable}, launch::ExecutionMode};
use crate::dsl::{
    frontend,
    post_processing::{checked_io::CheckedIoProcessor, saturating::SaturatingArithmeticProcessor, unroll::UnrollProcessor},
};

pub type WgslCompiler = ruda_compiler::wgsl::WgslCompiler<WgslLowering>;

#[derive(Clone, Copy, Debug, Default)]
pub struct WgslLowering;

impl ruda_compiler::wgsl::WgslLowering for WgslLowering {
    fn processors(mode: ExecutionMode, kernel_name: String, max_vector_size: usize) -> [Box<dyn Processor>; 3] {
        let checked_io: Box<dyn Processor> = Box::new(CheckedIoProcessor::new(mode, kernel_name));
        let unroll = Box::new(UnrollProcessor::new(max_vector_size));
        let saturating = Box::new(SaturatingArithmeticProcessor::new(true));
        [unroll, checked_io, saturating]
    }

    fn expand_erf(scope: &mut Scope, input: Variable, out: Variable) {
        frontend::expand_erf(scope, input, out);
    }

    fn expand_integer_power(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable) {
        frontend::expand_integer_power(scope, lhs, rhs, out);
    }

    fn expand_hypot(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable) {
        frontend::expand_hypot(scope, lhs, rhs, out);
    }

    fn expand_rhypot(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable) {
        frontend::expand_rhypot(scope, lhs, rhs, out);
    }

    fn expand_himul_64(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable) {
        frontend::expand_himul_64(scope, lhs, rhs, out);
    }

    fn expand_himul_sim(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable) {
        frontend::expand_himul_sim(scope, lhs, rhs, out);
    }
}
