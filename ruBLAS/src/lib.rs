//! Ruda BLAS operations and device tensor kernels.

#[cfg(feature = "cann")]
pub mod cann;

#[cfg(feature = "kernel-ir")]
#[allow(unsafe_code)]
pub mod kernel_ir;

#[cfg(feature = "tensor-vector")]
#[allow(unsafe_code)]
pub mod tensor_vector;

#[cfg(feature = "tensor-matmul")]
pub mod tensor_matmul;

#[cfg(feature = "tensor-int4")]
pub mod tensor_int4;

#[cfg(feature = "tensor-float-quant")]
pub mod tensor_float_quant;

#[cfg(feature = "tensor-nf4")]
#[allow(unsafe_code)]
pub mod tensor_nf4;

#[cfg(feature = "tensor-ggml")]
#[allow(unsafe_code)]
pub mod tensor_ggml;

#[cfg(feature = "tensor-grouped")]
pub mod tensor_grouped;
