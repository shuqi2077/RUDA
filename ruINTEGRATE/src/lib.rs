// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]
//! Host FP64 scientific integration. This package has no runtime, tensor, BLAS,
//! Python, or GPU dependency (the stiff method uses host rusolver). Callbacks are explicit host functions.
//!
//! Includes finite-interval Gauss-Kronrod 15/7 quadrature and Dormand-Prince 5(4)
//! integration of non-stiff real ODE systems. Errors are estimates, NOT rigorous
//! bounds. Additional explicit APIs implement BDF1, Hermite event location and
//! infinite-domain transforms. No hidden device readback or automatic CPU fallback.
mod error;
mod quadrature;
mod ode;
pub use error::IntegrationError;
pub use quadrature::{
    integrate, QuadratureOptions, QuadratureReport, QuadratureStatus
};
pub use ode::{
    solve_ivp, Rk45Options, OdeReport, OdeSample, OdeStatus
};
#[cfg(test)]mod tests;

mod improper;
mod stiff;
mod events;
pub use improper::{integrate_infinite,InfiniteInterval};
pub use stiff::{solve_bdf1,Bdf1Options,StiffReport,Jacobian};
pub use events::{solve_ivp_events,solve_bdf1_events,EventDirection,EventSpec,EventOptions,EventOccurrence,EventReport,StiffEventReport};
#[cfg(test)]mod advanced_tests;
