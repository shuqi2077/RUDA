mod base;
mod compare;
mod constructors;
mod conversion;
mod display;
mod error;
mod from;
mod initialize;
mod iteration;
mod storage;
mod tolerance;

pub use base::TensorData;
pub use error::DataError;
pub use tolerance::Tolerance;

#[cfg(test)]
mod tests;
