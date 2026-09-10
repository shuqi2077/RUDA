//! Quantization data representation.

// Re-exported types
pub use crate::quant::scheme::{
    BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};

/// Alignment (in bytes) for quantization parameters in serialized tensor data.
///
/// NOTE: This is currently f32-based since scales were originally always f32.
/// With `QuantParam` now supporting different precisions (F16, BF16, etc.),
/// this alignment may need to be revisited in the future.
pub const QPARAM_ALIGN: usize = core::mem::align_of::<f32>();

mod params;
mod packing;

#[cfg(feature = "tensor-data")]
mod bytes;

pub use params::*;
pub use packing::pack_i8s_to_u32s;
#[cfg(feature = "tensor-data")]
pub use bytes::QuantizedBytes;

#[cfg(test)]
mod tests;
