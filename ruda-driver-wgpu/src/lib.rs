#[macro_use]
extern crate derive_new;

extern crate alloc;

mod backend;
mod compilation;
mod execution;
mod memory;
mod device;
mod element;
mod graphics;
mod runtime;

pub use compilation::*;
pub use ruda_kernel::dsl::lowering::wgsl::WgslCompiler;
pub use execution::*;
pub use memory::*;
pub use device::*;
pub use element::*;
pub use graphics::*;
pub use runtime::*;

#[cfg(feature = "spirv")]
pub use backend::vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
pub use backend::metal;

#[cfg(all(test, not(feature = "spirv"), not(feature = "msl")))]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::WgpuRuntime;
    use half::f16;

    // Include 64-bit types (i64, u64) for WGSL as wgpu supports them. These don't exist on
    // native WebGPU however.
    //
    // Also include f16, this is an extension but supported by wgpu and WebGPU.
    ruda_kernel::dsl::testgen_all!(f32: [f16, f32], i32: [i32, i64], u32: [u32, u64]);
    ruda_kernel::library::testgen!();
    ruda_kernel::library::testgen_tensor_identity!([flex32, f32, u32]);
    ruda_kernel::library::testgen_quantized_view!(f32);
}

#[cfg(all(test, feature = "spirv"))]
#[allow(unexpected_cfgs)]
mod tests_spirv {
    pub type TestRuntime = crate::WgpuRuntime;
    use ruda_kernel::dsl::flex32;
    use half::f16;

    ruda_kernel::dsl::testgen_all!(f32: [f16, flex32, f32], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    ruda_kernel::library::testgen!();
    ruda_kernel::library::testgen_tensor_identity!([f16, flex32, f32, u32]);
    ruda_kernel::library::testgen_quantized_view!(f16);
}

#[cfg(all(test, feature = "msl"))]
#[allow(unexpected_cfgs)]
mod tests_msl {
    pub type TestRuntime = crate::WgpuRuntime;
    use half::f16;

    ruda_kernel::dsl::testgen_all!(f32: [f16, f32], i32: [i16, i32], u32: [u16, u32]);
    ruda_kernel::library::testgen!();
    ruda_kernel::library::testgen_tensor_identity!([f16, flex32, f32, u32]);
    ruda_kernel::library::testgen_quantized_view!(f16);
}
