use super::Subgroup;
use super::{ConstantArray, shader::ComputeShader};
use super::{Item, LocalArray, SharedArray};
use crate::wgsl::{self, SharedValue};

use ruda_core::{
    arguments::{Info, Metadata},
    backtrace::BackTrace,
    compiler::{CompilationError, WgpuCompilationOptions},
    ir::{self as ruda, Scope, StorageType, UIntKind},
    kernel,
    launch::ExecutionMode,
};
use super::WgslLowering;

pub const MAX_VECTOR_SIZE: usize = 4;

/// Wgsl Compiler.
#[derive(Clone, Default)]
pub struct WgslCompiler<P: WgslLowering> {
    lowering: core::marker::PhantomData<P>,
    kernel_name: String,
    info: Info,
    ext_meta_pos: Vec<u32>,
    local_invocation_index: bool,
    local_invocation_id: bool,
    // TODO: possible cleanup, this bool seems to not be used
    global_invocation_id: bool,
    workgroup_id: bool,
    subgroup_size: bool,
    subgroup_id: bool,
    num_subgroups: bool,
    subgroup_invocation_id: bool,
    id: bool,
    num_workgroups: bool,
    workgroup_id_no_axis: bool,
    workgroup_size_no_axis: bool,
    num_workgroup_no_axis: bool,
    shared_arrays: Vec<SharedArray>,
    shared_values: Vec<SharedValue>,
    const_arrays: Vec<ConstantArray>,
    local_arrays: Vec<LocalArray>,
    #[allow(dead_code)]
    compilation_options: WgpuCompilationOptions,
    strategy: ExecutionMode,
    subgroup_instructions_used: bool,
    f16_used: bool,
}

impl<P: WgslLowering> core::fmt::Debug for WgslCompiler<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WgslCompiler")
    }
}

impl<P: WgslLowering> ruda_core::compiler::Compiler for WgslCompiler<P> {
    type Representation = ComputeShader;
    type CompilationOptions = WgpuCompilationOptions;

    fn compile(
        &mut self,
        shader: kernel::KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
        address_type: StorageType,
    ) -> Result<Self::Representation, CompilationError> {
        self.compilation_options = *compilation_options;
        self.compile_shader(shader, mode, address_type)
    }

    fn elem_size(&self, elem: ruda::ElemType) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "wgsl"
    }
}

mod launch;
mod types;
mod scope;
mod arithmetic;
mod operators;

fn register_extensions(instructions: &[wgsl::Instruction]) -> Vec<wgsl::Extension> {
    let mut extensions = Vec::new();

    let mut register_extension = |extension: wgsl::Extension| {
        if !extensions.contains(&extension) {
            extensions.push(extension);
        }
    };

    // Since not all instructions are native to WGSL, we need to add the custom ones.
    for instruction in instructions {
        match instruction {
            wgsl::Instruction::Powf { lhs: _, rhs, out } => {
                register_extension(wgsl::Extension::PowfPrimitive(out.elem()));
                register_extension(wgsl::powf_extension(rhs, out));
            }
            #[cfg(target_os = "macos")]
            wgsl::Instruction::Tanh { input, out: _ } => {
                register_extension(wgsl::Extension::SafeTanhPrimitive(input.elem()));
                register_extension(wgsl::Extension::SafeTanh(input.item()));
            }
            wgsl::Instruction::IsNan { input, out } => {
                register_extension(wgsl::Extension::IsNanPrimitive(input.elem()));
                register_extension(wgsl::Extension::IsNan(input.item(), out.item()));
            }
            wgsl::Instruction::IsInf { input, out } => {
                register_extension(wgsl::Extension::IsInfPrimitive(input.elem()));
                register_extension(wgsl::Extension::IsInf(input.item(), out.item()));
            }
            wgsl::Instruction::If { instructions, .. } => {
                for extension in register_extensions(instructions) {
                    register_extension(extension);
                }
            }
            wgsl::Instruction::IfElse {
                instructions_if,
                instructions_else,
                ..
            } => {
                for extension in register_extensions(instructions_if) {
                    register_extension(extension);
                }
                for extension in register_extensions(instructions_else) {
                    register_extension(extension);
                }
            }
            wgsl::Instruction::Loop { instructions } => {
                for extension in register_extensions(instructions) {
                    register_extension(extension);
                }
            }
            wgsl::Instruction::RangeLoop { instructions, .. } => {
                for extension in register_extensions(instructions) {
                    register_extension(extension);
                }
            }
            _ => {}
        }
    }

    extensions
}
