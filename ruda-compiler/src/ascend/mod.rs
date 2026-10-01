//! Common `ruda_core::kernel::KernelDefinition` -> Ascend vector device code.
//!
//! Checked subsets: independent FP32 maps, or explicit 32-lane row kernels with
//! affine indexing and Plane::Sum/Max. Unsupported operations never fall back to
//! an external library. All formulas remain common Rust IR; Bisheng compiles the
//! generated CCE intermediate. This is not a direct Rust-to-Ascend-ISA backend.
mod lower;
mod emit;
mod plan;
mod rows;
mod row_emit;
pub mod row_programs;
#[cfg(test)] mod row_tests;
pub mod programs;
#[cfg(test)] mod tests;

use ruda_core::{backtrace::BackTrace, compiler::{CompilationError, Compiler},
    ir::{ElemType, StorageType, UIntKind}, kernel::{KernelDefinition, Visibility},
    launch::ExecutionMode};
use std::fmt;
pub(crate) type Result<T> = std::result::Result<T, CompilationError>;
pub(crate) fn invalid(reason: impl Into<String>) -> CompilationError {
    CompilationError::Validation { reason: format!("Ascend CCE: {}", reason.into()), backtrace: BackTrace::capture() }
}
pub(crate) fn unsupported(reason: impl Into<String>) -> CompilationError {
    CompilationError::UnsupportedInstruction { reason: format!("Ascend CCE: {}", reason.into()), backtrace: BackTrace::capture() }
}

/// Compile target, not a hardware certification. No target guessed from the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AscendTarget { Ascend950DT }
impl AscendTarget {
    pub fn soc(self) -> &'static str { "Ascend950DT" }
    pub fn arch(self) -> &'static str { "dav-c310" }
}

/// Shape-specialized lowering. `elements` is the complete logical element domain.
/// `tile_elements` configures map mode only; row mode allocates a complete row.
/// Launches use the returned physical `block_dim`, NOT the original GPU thread grid.
#[derive(Clone, Debug)]
pub struct AscendOptions {
    pub target: Option<AscendTarget>,
    pub elements: u64,
    /// Some(width): one logical 32-lane plane owns one complete row.
    /// None preserves the independent-element map compiler.
    pub row_width: Option<u32>,
    pub tile_elements: u32,
    pub vector_cores: u32,
    pub ub_limit_bytes: u32,
    pub reuse_temporaries: bool,
}
impl Default for AscendOptions {
    fn default() -> Self { Self { target: None, elements: 0, row_width: None, tile_elements: 256,
        vector_cores: 32, ub_limit_bytes: 128 * 1024, reuse_temporaries: true } }
}
impl AscendOptions {
    fn validate(&self) -> Result<AscendTarget> {
        let target = self.target.ok_or_else(|| invalid("explicit Ascend target required"))?;
        if self.elements > u32::MAX as u64 { return Err(invalid("logical element count exceeds u32")); }
        if !(8..=4096).contains(&self.tile_elements) || self.tile_elements % 8 != 0 {
            return Err(invalid("tile_elements must be 8..4096 and a multiple of 8 FP32 elements"));
        }
        if !(1..=32).contains(&self.vector_cores) { return Err(invalid("vector_cores must be 1..32")); }
        if self.ub_limit_bytes == 0 || self.ub_limit_bytes > 128 * 1024 {
            return Err(invalid("UB limit must be 1..131072 bytes; larger limits are not certified"));
        }
        Ok(target)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AscendBinding { pub id: u32, pub writable: bool, pub bytes: u64 }

/// Immutable compiler result. Only the compiler can create its binding contract.
/// Buffers follow original KernelDefinition order, with no hidden runtime metadata.
#[derive(Clone, Debug)]
pub struct AscendKernel {
    source: String, entrypoint: String, target: AscendTarget,
    elements: u64, row_width: Option<u32>, block_dim: u32, tile_elements: u32,
    ub_bytes: u32, temporary_slots: usize, bindings: Vec<AscendBinding>,
}
impl AscendKernel {
    pub fn source(&self) -> &str { &self.source }
    pub fn entrypoint(&self) -> &str { &self.entrypoint }
    pub fn target(&self) -> AscendTarget { self.target }
    pub fn elements(&self) -> u64 { self.elements }
    pub fn row_width(&self) -> Option<u32> { self.row_width }
    pub fn block_dim(&self) -> u32 { self.block_dim }
    pub fn tile_elements(&self) -> u32 { self.tile_elements }
    pub fn ub_bytes(&self) -> u32 { self.ub_bytes }
    pub fn temporary_slots(&self) -> usize { self.temporary_slots }
    pub fn bindings(&self) -> &[AscendBinding] { &self.bindings }
    /// Build-time metadata only. Not a loadable artifact until a real CANN build
    /// succeeds and the build tool adds source/object/compiler digests.
    pub fn build_contract(&self) -> String {
        let schema = if self.row_width.is_some() { "ruda.ascend.common-row.v1" } else { "ruda.ascend.common-map.v1" };
        let mut t = format!("schema={schema}\nsource_language=ruda-kernel-ir\nlowering=ascendc-vector\nsoc={}\narch={}\nkernel_name={}\nelements={}\nblock_dim={}\ntile_elements={}\nub_bytes={}\nbindings={}\n",
            self.target.soc(), self.target.arch(), self.entrypoint, self.elements,
            self.block_dim, self.tile_elements, self.ub_bytes, self.bindings.len());
        if let Some(width) = self.row_width { t.push_str(&format!("row_width={width}\nlogical_plane=32\n")); }
        for (i, b) in self.bindings.iter().enumerate() {
            t.push_str(&format!("binding_{i}={},{},{}\n", b.id, if b.writable {"w"} else {"r"}, b.bytes));
        }
        t
    }
}
impl fmt::Display for AscendKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.source) }
}

#[derive(Clone, Debug, Default)]
pub struct AscendCompiler;
impl AscendCompiler { pub const CACHE_VERSION: u32 = 2; }
impl Compiler for AscendCompiler {
    type Representation = AscendKernel;
    type CompilationOptions = AscendOptions;
    fn compile(&mut self, kernel: KernelDefinition, o: &AscendOptions,
        mode: ExecutionMode, address: StorageType) -> Result<AscendKernel> {
        let target = o.validate()?;
        if mode != ExecutionMode::Checked {
            return Err(unsupported("only Checked mode; Validate diagnostics and Unchecked mode are not implemented"));
        }
        if address != StorageType::from(UIntKind::U64) { return Err(unsupported("64-bit addresses required")); }
        if let Some(width) = o.row_width {
            let p = rows::lower(kernel, o.elements, width)?;
            let alloc = rows::allocate(&p, o.reuse_temporaries, o.ub_limit_bytes)?;
            let source = row_emit::emit(&p, &alloc);
            let bindings = p.bindings.iter().map(|b| AscendBinding { id: b.arg.id,
                writable: b.arg.visibility == Visibility::ReadWrite,
                bytes: b.kind.count(p.rows, width) * 4 }).collect();
            return Ok(AscendKernel { source, entrypoint: p.name, target, elements: o.elements,
                row_width: Some(width), block_dim: o.vector_cores, tile_elements: width,
                ub_bytes: alloc.ub_bytes, temporary_slots: alloc.slots, bindings });
        }
        let p = lower::lower(kernel, o.elements)?;
        let alloc = plan::allocate(&p, o.reuse_temporaries)?;
        let vectors = p.bindings.len().checked_add(alloc.slots).ok_or_else(|| invalid("UB count overflow"))?;
        let ub = vectors.checked_mul(o.tile_elements as usize).and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("UB byte count overflow"))?;
        if ub > o.ub_limit_bytes as usize { return Err(unsupported(format!("kernel requires {ub} UB bytes, limit is {}", o.ub_limit_bytes))); }
        let source = emit::emit(&p, &alloc, o);
        Ok(AscendKernel { source, entrypoint: p.name, target, elements: o.elements, row_width: None,
            block_dim: o.vector_cores, tile_elements: o.tile_elements, ub_bytes: ub as u32,
            temporary_slots: alloc.slots,
            bindings: p.bindings.iter().map(|b| AscendBinding { id: b.id,
                writable: b.visibility == Visibility::ReadWrite, bytes: o.elements * 4 }).collect() })
    }
    fn elem_size(&self, elem: ElemType) -> usize { elem.size() }
    fn extension(&self) -> &'static str { "asc" }
}
