use alloc::string::String;
use crate::dsl::prelude::*;
use ruda_kernel_macros::intrinsic;

#[ruda]
#[allow(unused_variables)]
/// Push a validation error that will make the kernel compilation to fail.
///
/// # Notes
///
/// The error can be caught after the kernel is launched.
pub fn push_validation_error(#[comptime] msg: String) {
    intrinsic! {|scope| scope.push_error(msg)}
}
