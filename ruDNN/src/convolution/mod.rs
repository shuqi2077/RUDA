pub mod components;
pub mod definition;
pub mod kernels;
pub mod launch;
pub mod routines;

#[cfg(feature = "cpu-reference")]
pub mod cpu_reference;

// Re-export per-operation modules at the crate root for internal paths
// (`crate::convolution::forward`, etc.) and for downstream users that previously relied on
// `rudnn::convolution::*`.
pub use kernels::{backward_data, backward_weight, forward};

// Top-level launcher: the single public entry point.
pub use launch::{
    AcceleratedTileKind, ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy, launch_ref,
};

#[cfg(feature = "tensor-convolution")]
pub mod tensor;
