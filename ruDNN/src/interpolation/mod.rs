mod base;
pub(crate) mod bicubic;
mod bicubic_backward;
mod bilinear;
mod bilinear_backward;
mod lanczos3;
mod lanczos3_backward;
mod nearest;
mod nearest_backward;

pub use base::*;
