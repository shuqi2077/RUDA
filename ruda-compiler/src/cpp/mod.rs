pub mod shared;

pub use shared::ComputeKernel;
pub use shared::register_supported_types;
pub use shared::{Dialect, DialectWmmaCompiler};

/// Format CPP code.
pub mod formatter;

//#[cfg(feature = "cuda")]
pub mod cuda;
//#[cfg(feature = "hip")]
pub mod hip;
#[cfg(feature = "cpp-metal")]
pub mod metal;

