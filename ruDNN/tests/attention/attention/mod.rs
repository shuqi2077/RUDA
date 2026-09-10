pub mod basic;
#[cfg(feature = "extended")]
pub mod extended;

pub(crate) mod launcher;

pub(crate) use rudnn::attention::kernel_ir::cpu_reference::assert_result;
