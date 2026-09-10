#[cfg(feature = "quantization-kernels")]
pub mod dequantize;

#[cfg(feature = "quantization-kernels")]
pub mod quantize;

#[cfg(feature = "quantization-kernels")]
pub mod layout;

pub use ruda_core::quant::scheme;

#[cfg(feature = "quantization-kernels")]
pub(crate) mod utils {
    use crate::quantization::scheme::{QuantLevel, QuantScheme};

    pub(crate) fn check_block_size_compat(scheme: &QuantScheme, div: usize) {
        // Validate block size compatibility
        if let QuantScheme {
            level: QuantLevel::Block(block_size),
            ..
        } = scheme
        {
            let block_size = *block_size.as_slice().last().unwrap() as usize;
            assert!(
                block_size.is_multiple_of(div),
                "Block size must be divisible by {div}, got block_size={block_size}"
            );
        }
    }
}
