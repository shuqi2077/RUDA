use super::{
    BinaryInstruction, Body, Component, ComputeKernel, ConstArray, Dialect, Elem, FP4Kind, FP6Kind,
    FP8Kind, Fragment, FragmentIdent, FragmentLayout, IndexAssignInstruction, IndexInstruction,
    Instruction, Item, KernelArg, LocalArray, SharedMemory, UnaryInstruction, Variable,
    WarpInstruction, WmmaInstruction, barrier::{self, BarrierOps}, pipeline::PipelineOps,
};
use crate::cpp::shared::MmaShape;
use ruda_core::backtrace::BackTrace;
use ruda_core::{
    launch::{RudaDim, ExecutionMode},
    ir::{
        self as gpu, DeviceProperties, ElemType, FloatKind, InstructionModes, OpaqueType,
        Operation, Processor, SourceLoc, StorageType, Type,
        features::{AtomicUsage, EnumSet, TypeUsage},
    },
    ir::FastMath,
    kernel::KernelDefinition,
};
use crate::optimizer::{Optimizer, SharedLiveness};
use ruda_core::compiler::{CompilationError, Compiler};
use std::{collections::HashSet, fmt::Debug};

pub(super) static COUNTER_TMP_VAR: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

#[derive(Clone, Debug)]
pub struct CompilationOptions {
    pub warp_size: u32,
    pub supports_features: CppSupportedFeatures,
}

#[derive(Clone, Debug, Default)]
pub struct CppSupportedFeatures {
    pub grid_constants: bool,
    pub clusters: bool,
    pub fast_math: bool,
    pub fast_tanh: bool,
    pub elect_sync: bool,
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self {
            warp_size: 32,
            supports_features: Default::default(),
        }
    }
}

/// Ruda indexes flags.
/// When true the corresponding index is declared and computed as needed in the kernel.
#[derive(Debug, Clone, Default)]
pub struct RudaIndexFlags {
    pub absolute_pos: bool,
    pub absolute_pos_tuple: bool,
    pub ruda_count: bool,
    pub ruda_count_tuple: bool,
    pub ruda_dim: bool,
    pub ruda_dim_tuple: bool,
    pub ruda_pos: bool,
    pub ruda_pos_tuple: bool,
    pub plane_dim: bool,
    pub plane_dim_checked: bool,
    pub plane_pos: bool,
    pub plane_count: bool,
    pub unit_pos: bool,
    pub unit_pos_tuple: bool,
    pub unit_pos_plane: bool,
    pub cluster_pos: bool,
}

/// Flags gathered during Ruda IR translation for the kernel compilation.
#[derive(Debug, Clone)]
pub struct Flags<D: Dialect> {
    pub elem_fp4: bool,
    pub elem_fp6: bool,
    pub elem_fp8: bool,
    pub elem_bf16: bool,
    pub elem_f16: bool,
    pub elem_tf32: bool,
    pub indexes: RudaIndexFlags,
    pub op_barrier: bool,
    pub op_pipeline: bool,
    pub inst_tma: bool,
    pub inst_tma_im2col: bool,
    pub inst_wmma: bool,
    pub inst_ptx_wrappers: bool,
    pub inst_async_copy: bool,
    pub use_grid_constants: bool,
    pub static_meta_length: usize,
    pub has_dynamic_meta: bool,
    pub has_info: bool,
    pub ruda_dim: RudaDim,
    pub cluster_dim: Option<RudaDim>,
    pub address_type: Item<D>,
}

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Debug)]
pub struct CppCompiler<D: Dialect, P: super::DialectProcessors> {
    processors: core::marker::PhantomData<P>,
    kernel_name: String,
    barriers: Vec<BarrierOps<D>>,
    compilation_options: CompilationOptions,
    const_arrays: Vec<ConstArray<D>>,
    ext_meta_positions: Vec<u32>,
    cluster_dim: RudaDim,
    extensions: Vec<D::Extension>,
    flags: Flags<D>,
    items: HashSet<Item<D>>,
    local_arrays: Vec<LocalArray<D>>,
    info: ruda_core::arguments::Info,
    pipelines: Vec<PipelineOps<D>>,
    source_loc: Option<SourceLoc>,
    strategy: ExecutionMode,
    addr_type: Item<D>,
}

impl<D: Dialect> Default for Flags<D> {
    fn default() -> Self {
        Self {
            elem_fp4: Default::default(),
            elem_fp6: Default::default(),
            elem_fp8: Default::default(),
            elem_bf16: Default::default(),
            elem_f16: Default::default(),
            elem_tf32: Default::default(),
            indexes: Default::default(),
            op_barrier: Default::default(),
            op_pipeline: Default::default(),
            inst_tma: Default::default(),
            inst_tma_im2col: Default::default(),
            inst_wmma: Default::default(),
            inst_ptx_wrappers: Default::default(),
            inst_async_copy: Default::default(),
            use_grid_constants: Default::default(),
            static_meta_length: Default::default(),
            has_info: Default::default(),
            has_dynamic_meta: Default::default(),
            ruda_dim: RudaDim::new_single(),
            cluster_dim: Default::default(),
            address_type: Item::scalar(Elem::U32, true),
        }
    }
}

impl<D: Dialect, P: super::DialectProcessors> Default for CppCompiler<D, P> {
    fn default() -> Self {
        Self {
            processors: core::marker::PhantomData,
            kernel_name: Default::default(),
            barriers: Default::default(),
            compilation_options: Default::default(),
            const_arrays: Default::default(),
            ext_meta_positions: Default::default(),
            cluster_dim: RudaDim::new_single(),
            extensions: Default::default(),
            flags: Flags::default(),
            items: Default::default(),
            local_arrays: Default::default(),
            info: Default::default(),
            pipelines: Default::default(),
            source_loc: Default::default(),
            strategy: Default::default(),
            addr_type: Item::scalar(Elem::U32, true),
        }
    }
}

impl<D: Dialect, P: super::DialectProcessors> Compiler for CppCompiler<D, P> {
    type Representation = ComputeKernel<D>;
    type CompilationOptions = CompilationOptions;

    fn compile(
        &mut self,
        mut kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        strategy: ExecutionMode,
        addr_type: StorageType,
    ) -> Result<Self::Representation, CompilationError> {
        let errors = kernel.body.pop_errors();
        if !errors.is_empty() {
            let mut reason = "Can't compile cpp kernel\nCaused by:\n  ".to_string();
            for error in errors {
                reason += error.as_str();
                reason += "\n";
            }

            return Err(CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            });
        }

        self.addr_type = self.compile_type(addr_type.into());
        self.compilation_options = compilation_options.clone();
        self.strategy = strategy;
        self.kernel_name = kernel.options.kernel_name.clone();

        if !self.compilation_options.supports_features.clusters {
            kernel.options.cluster_dim = None;
        }
        self.cluster_dim = kernel.options.cluster_dim.unwrap_or(RudaDim::new_single());

        let ir = self.clone().compile_ir(kernel, addr_type);
        COUNTER_TMP_VAR.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(ir)
    }

    fn elem_size(&self, elem: gpu::ElemType) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "cpp"
    }
}

mod kernel;
mod scope;
mod instructions;
mod arithmetic;
mod types;

fn is_fp4_fp6_fp8(elem: gpu::ElemType) -> bool {
    match elem {
        gpu::ElemType::Float(kind) => matches!(
            kind,
            FloatKind::E2M1
                | FloatKind::E2M3
                | FloatKind::E3M2
                | FloatKind::E4M3
                | FloatKind::E5M2
                | FloatKind::UE8M0
        ),
        _ => false,
    }
}

fn const_u32<D: Dialect>(value: u32) -> Variable<D> {
    Variable::Constant(
        gpu::ConstantValue::UInt(value as u64),
        Item::new(Elem::U32, 1, true),
    )
}

pub fn register_supported_types(props: &mut DeviceProperties) {
    props.register_address_type(gpu::AddressType::U32);
    props.register_address_type(gpu::AddressType::U64);

    let supported_types = [
        gpu::ElemType::UInt(gpu::UIntKind::U8),
        gpu::ElemType::UInt(gpu::UIntKind::U16),
        gpu::ElemType::UInt(gpu::UIntKind::U32),
        gpu::ElemType::UInt(gpu::UIntKind::U64),
        gpu::ElemType::Int(gpu::IntKind::I8),
        gpu::ElemType::Int(gpu::IntKind::I16),
        gpu::ElemType::Int(gpu::IntKind::I32),
        gpu::ElemType::Int(gpu::IntKind::I64),
        gpu::ElemType::Float(gpu::FloatKind::BF16),
        gpu::ElemType::Float(gpu::FloatKind::F16),
        gpu::ElemType::Float(gpu::FloatKind::F32),
        gpu::ElemType::Float(gpu::FloatKind::Flex32),
        // Causes CUDA_ERROR_INVALID_VALUE for matmul, disabling until that can be investigated
        //gpu::Elem::Float(gpu::FloatKind::F64),
        gpu::ElemType::Bool,
    ];

    let supported_atomic_types = [
        gpu::ElemType::Int(gpu::IntKind::I32),
        gpu::ElemType::Int(gpu::IntKind::I64),
        gpu::ElemType::UInt(gpu::UIntKind::U32),
        gpu::ElemType::UInt(gpu::UIntKind::U64),
        gpu::ElemType::Float(gpu::FloatKind::F32),
    ];

    for ty in supported_types {
        props.register_type_usage(ty, TypeUsage::all());
    }

    for ty in supported_atomic_types {
        props.register_atomic_type_usage(
            Type::new(gpu::StorageType::Atomic(ty)),
            AtomicUsage::Add | AtomicUsage::LoadStore,
        );
    }
}
