/// Re-export for testgen macros.
pub use test_log;

pub mod event;
pub mod reinterpret_slice;
pub mod tensor;
pub mod trigonometry;
pub mod view;

#[macro_export]
macro_rules! testgen {
    () => {
        mod test_ruda_std {
            use super::*;
            use half::{bf16, f16};

            ruda_kernel::library::testgen_reinterpret_slice!();
            ruda_kernel::library::testgen_trigonometry!();
            ruda_kernel::library::testgen_event!();
        }
    };
}
