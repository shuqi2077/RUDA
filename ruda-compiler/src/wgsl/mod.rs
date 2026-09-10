mod base;
mod body;
mod compiler;
mod extension;
mod instructions;
pub(crate) mod shader;
mod subgroup;

pub use base::*;
pub use body::*;
pub use compiler::*;
pub use extension::*;
pub use instructions::*;
pub use shader::*;
pub use subgroup::*;

mod lowering;
pub use lowering::WgslLowering;
