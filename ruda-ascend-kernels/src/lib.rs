//! Rust-authored, staged device code, NOT an FFI wrapper around DeepGEMM kernels.
//!
//! `kernel::build` owns scheduling, addressing, the reduction loop and pipe events.
//! The target printer lowers the resulting program to CANN intrinsic calls. The
//! generated CCE source is an intermediate representation, not a handwritten or
//! included DeepGEMM implementation. Bisheng is still required to make NPU code.
//! Only the direct-store BF16 subset is ported. No FP8/FP4/MQA/MegaMoE claim.
#![forbid(unsafe_code)]
pub mod config;
pub mod ir;
pub mod layout;
pub mod scheduler;
pub mod kernel;
pub mod ascend;
#[cfg(test)]
mod tests;

pub use config::{Kind, Major, Output, Spec};
pub fn emit(spec: Spec) -> Result<String, String> {
    spec.check()?;
    let program = kernel::build(spec)?;
    program.check()?;
    ascend::emit(&program)
}
