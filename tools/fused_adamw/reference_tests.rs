// SPDX-License-Identifier: Apache-2.0
// Standalone reference tests: no Cargo registry, GPU, or other RUDA crate needed.
// rustc --edition 2024 --test tools/fused_adamw/reference_tests.rs -o ...
#[path = "../../ruda-optim/src/fused_adamw/config.rs"]
mod config;
pub use config::{AdamWOptions, FusedAdamWError, StepCoefficients, StepControl};
#[path = "../../ruda-optim/src/fused_adamw/reference.rs"]
mod reference;
