#[allow(unused_imports)]
#[macro_use]
extern crate derive_new;
extern crate alloc;

pub mod execution;
pub mod device;
pub mod runtime;
pub use device::*;
pub use runtime::HipRuntime;

#[cfg(not(feature = "rocwmma"))]
pub(crate) type HipWmmaCompiler = ruda_compiler::cpp::hip::mma::WmmaIntrinsicCompiler;

#[cfg(feature = "rocwmma")]
pub(crate) type HipWmmaCompiler = ruda_compiler::cpp::hip::mma::RocWmmaCompiler;

#[cfg(test)]
mod tests {
    use half::f16;
    pub type TestRuntime = crate::HipRuntime;

    ruda_kernel::library::testgen!();
    ruda_kernel::dsl::testgen_all!(f32: [f16, f32], i32: [i16, i32], u32: [u16, u32]);
}
