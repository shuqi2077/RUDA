pub mod barrier;
pub mod branch;
pub mod cmma;
pub mod synchronization;
pub mod unaligned_vector;

/// Module containing compile-time information about the current runtime.
pub mod comptime;

mod base;
pub mod comptime_error;
mod comptime_option;
mod const_expand;
mod container;
mod debug;
mod element;
mod indexation;
mod integer_power;
mod list;
mod operation;
mod options;
mod plane;
mod polyfills;
mod runtime_option;
mod scalar;
mod topology;
mod trigonometry;
mod validation;

pub use branch::*;
pub use comptime_option::*;
pub use const_expand::*;
pub use container::*;
pub use debug::*;
pub use element::*;
pub use indexation::*;
pub use integer_power::expand_integer_power;
pub use list::*;
pub use operation::*;
pub use options::*;
pub use plane::*;
pub use polyfills::*;
pub use runtime_option::*;
pub use scalar::*;
pub use synchronization::*;
pub use topology::*;
pub use trigonometry::*;
pub use validation::*;

pub use crate::dsl::{debug_print, debug_print_expand};
