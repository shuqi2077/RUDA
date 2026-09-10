mod operation;
mod tensor;

pub use operation::{CumulativeOp, CumulativeOpFamily};
pub use tensor::{cumsum, cumprod, cummin, cummax};
