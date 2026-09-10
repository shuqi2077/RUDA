

#[cfg(feature = "kernel-ir")]
pub mod kernel_ir;

#[cfg(feature = "attention-fallback")]
pub mod fallback;
#[cfg(feature = "attention-fallback")]
pub mod fallback_ops;
#[cfg(feature = "tensor-attention")]
pub mod tensor;
