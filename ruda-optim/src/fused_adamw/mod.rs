// SPDX-License-Identifier: Apache-2.0
//! Opt-in, out-of-place AdamW updates with FP32 master parameters and moments.
//!
//! `fused-adamw` provides configuration and a CPU **reference**, not a fallback.
//! `fused-adamw-device` adds a single-kernel device path with F32/F16/BF16 gradients.
//! Existing [`crate::AdamW`] dispatch and checkpoint formats are unchanged.
//!
//! This is a low-level optimizer primitive, not a differentiable optimizer or an
//! automatic replacement for the model optimizer adaptor. See
//! `docs/en/fused-adamw.md` for the numerical, asynchronous and benchmark contract.

mod config;
pub use config::{AdamWOptions, FusedAdamWError, StepCoefficients, StepControl};

/// An independent, explicitly invoked CPU numerical oracle for tests.
pub mod reference;

#[cfg(feature = "fused-adamw-device")]
mod kernel;
#[cfg(feature = "fused-adamw-device")]
mod device;
#[cfg(feature = "fused-adamw-device")]
pub use device::{AdamWState, AdamWUpdate, adamw_step};

#[cfg(test)]
mod existing_optimizer_tests;
