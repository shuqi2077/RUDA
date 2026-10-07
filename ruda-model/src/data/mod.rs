/// Dataloader module.
#[cfg(feature = "dataset")]
pub mod dataloader;

/// Committed sample cursors and explicit topology-changing epoch continuation.
#[cfg(feature = "std")]
pub mod sampler;

/// Token-budget batches retaining committed distributed sample order.
#[cfg(feature = "std")]
pub mod token_batch;

/// Explicit-label causal-LM examples and packed/padded device collation.
#[cfg(feature = "std")]
pub mod causal;

/// Dataset module.
#[cfg(feature = "dataset")]
pub mod dataset {
    pub use ruda_dataset::*;
}

/// Network module.
#[cfg(feature = "network")]
pub mod network {
    pub use ruda_io::network::*;
}
