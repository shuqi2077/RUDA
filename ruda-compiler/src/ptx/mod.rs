//! Direct Kernel IR to PTX compilation, independent of CUDA C++ and NVRTC.

mod emit;
pub mod row;
mod bitwise;
mod barrier;
mod atomic;
mod arrays;
mod register_arrays;
mod memory_copy;
mod memory_scalar;
mod native_memory;
mod matrix;
mod diagnostics;
mod printf;
mod inline_asm;
mod validation;
mod control_flow;
mod cluster;
mod integer;
mod constant_division;
mod conversion;
mod predicates;
mod plane;
mod numeric;
mod fast_math;
mod math;
mod binary_math;
mod double_math;
mod trigonometry;
mod exp;
mod expm1;
mod log;
mod log1p;
mod tanh;
mod atanh;
mod inverse_hyperbolic;
mod hyperbolic;
mod power;
mod remainder;
mod erf;
mod erf_f64;
mod vector;
mod half;
mod fp8;
mod packed;
mod metadata;
mod operations;
mod shared;
mod types;

use ruda_core::{
    backtrace::BackTrace,
    compiler::{CompilationError, Compiler},
    ir::{ElemType, StorageType},
    kernel::KernelDefinition,
    launch::{RudaDim, ExecutionMode},
};
use std::fmt;

type Result<T> = std::result::Result<T, CompilationError>;

fn unsupported(reason: impl Into<String>) -> CompilationError {
    CompilationError::UnsupportedInstruction {
        reason: format!("Direct PTX: {}", reason.into()),
        backtrace: BackTrace::capture(),
    }
}

fn invalid(reason: impl Into<String>) -> CompilationError {
    CompilationError::Validation {
        reason: format!("Direct PTX: {}", reason.into()),
        backtrace: BackTrace::capture(),
    }
}

/// Target selected by the caller for its device and installed driver.
#[derive(Clone, Copy, Debug)]
pub struct PtxTarget {
    pub version: (u32, u32),
    pub sm: u32,
}

/// No device architecture is inferred from the host running the compiler.
#[derive(Clone, Debug, Default)]
pub struct PtxCompilationOptions {
    pub target: Option<PtxTarget>,
}

/// Compiled PTX and the launch properties of the original kernel.
#[derive(Clone, Debug)]
pub struct PtxKernel {
    pub io: ruda_core::kernel::KernelIo,
    pub source: String,
    pub entrypoint: String,
    pub ruda_dim: RudaDim,
    pub shared_memory_bytes: usize,
    /// Position of the dynamic metadata pointer, when required by the argument ABI.
    pub dynamic_metadata_index: Option<usize>,
}

impl fmt::Display for PtxKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.source)
    }
}

/// Selectable direct backend. Unsupported IR is an error, never an NVRTC fallback.
#[derive(Clone, Debug, Default)]
pub struct PtxCompiler;

impl PtxCompiler {
    pub const CACHE_VERSION: u32 = 61;
}

impl Compiler for PtxCompiler {
    type Representation = PtxKernel;
    type CompilationOptions = PtxCompilationOptions;

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        options: &Self::CompilationOptions,
        mode: ExecutionMode,
        address_type: StorageType,
    ) -> Result<PtxKernel> {
        let target = options
            .target
            .ok_or_else(|| invalid("explicit PTX version and SM target required"))?;
        let name = kernel.options.kernel_name.clone();
        emit::Emitter::compile(kernel, target, mode, address_type).map_err(|error| {
            diagnostics::context(error, format!("kernel {name:?}, sm_{}, PTX {}.{}, mode {mode:?}, address {address_type:?}", target.sm, target.version.0, target.version.1))
        })
    }

    fn elem_size(&self, elem: ElemType) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "ptx"
    }

    fn buffer_io(&self, repr: &Self::Representation) -> Option<ruda_core::kernel::KernelIo> {
        Some(repr.io.clone())
    }
}

#[cfg(test)]
mod tests;
