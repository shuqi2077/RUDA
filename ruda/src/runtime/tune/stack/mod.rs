//! Opt-in, shared autotuning above individual kernels.
//!
//! Tensor operators and fused graphs use the same controller as explicit application plans.
//! Existing LocalTuner behavior is unchanged until enable_stack_autotune is called; only sets
//! declaring an exact workload signature and reference participate in the new path.
#![forbid(unsafe_code)]
mod cache;
mod engine;
mod policy;
mod runtime_adapter;
/// Non-cryptographic content fingerprint for workload/dependency ids; not anonymization.
pub use cache::digest as workload_digest;
pub use engine::*;
pub use policy::*;
pub use runtime_adapter::*;

#[cfg(test)]
mod tests;
