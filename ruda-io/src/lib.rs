#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! Host-side input/output utilities for Ruda.

/// Network download utilities.
#[cfg(feature = "network")]
pub mod network;
