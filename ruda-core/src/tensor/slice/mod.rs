//! Tensor slice utilities.

#[macro_use]
mod macros;

mod arguments;
mod base;
mod iterator;
mod parse;
mod ranges;
mod shape_ops;

pub use arguments::SliceArg;
pub use base::Slice;
pub use iterator::SliceIter;
pub use shape_ops::SliceOps;

#[cfg(test)]
mod tests;
