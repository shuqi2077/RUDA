
use ruda_core::backtrace::BackTrace;
use ruda::runtime::compiler::CompilationError;
use ruda_compiler::mlir::shared_memories::SharedMemories;
pub use ruda_compiler::mlir::register_supported_types;

use ruda_kernel::dsl::{
    Compiler,
    ir::{self, StorageType},
    prelude::KernelDefinition,
    server::ExecutionMode,
};
use crate::execution::jit::MlirEngine;


#[derive(Clone, Debug, Default)]
pub struct MlirCompiler {}

#[derive(Default, Debug)]
pub struct MlirCompilerOptions {}

impl Compiler for MlirCompiler {
    type Representation = MlirEngine;

    type CompilationOptions = MlirCompilerOptions;

    fn compile(
        &mut self,
        mut kernel: KernelDefinition,
        _compilation_options: &Self::CompilationOptions, // TODO pass this through the visitor, though it doesn't need anything for the moment
        mode: ExecutionMode, // TODO support this by adding array bound checking
        addr_type: StorageType,
    ) -> Result<Self::Representation, CompilationError> {
        let errors = kernel.body.pop_errors();
        if !errors.is_empty() {
            let mut reason = "Can't compile mlir kernel".to_string();
            for error in errors {
                reason += error.as_str();
                reason += "\n";
            }

            return Err(CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            });
        }

        #[cfg(feature = "mlir-dump")]
        dump_scope(&kernel.body, &kernel.options.kernel_name);
        let opt = ruda_kernel::dsl::lowering::mlir::optimize(&kernel, mode);

        let mut shared_memories = SharedMemories::default();
        shared_memories.visit(&opt);

        #[cfg(feature = "mlir-dump")]
        dump_opt(&opt, &kernel.options.kernel_name);
        Ok(MlirEngine::from_ruda_ir(
            kernel,
            &opt,
            shared_memories,
            addr_type,
        ))
    }

    fn elem_size(&self, elem: ir::ElemType) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "mlir"
    }
}

#[cfg(feature = "mlir-dump")]
fn dump_scope(scope: &ruda_kernel::dsl::prelude::Scope, name: &str) {
    use std::fs;

    if let Ok(dir) = std::env::var("RUDA_DEBUG_MLIR") {
        let path = format!("{dir}/{name}");
        let _ = fs::create_dir(&path);
        fs::write(format!("{path}/ruda.ir.txt"), format!("{}", scope)).unwrap();
    }
}

#[cfg(feature = "mlir-dump")]
fn dump_opt(opt: &ruda_compiler::optimizer::Optimizer, name: &str) {
    if let Ok(dir) = std::env::var("RUDA_DEBUG_MLIR") {
        use std::fs;
        let path = format!("{dir}/{name}");
        let _ = fs::create_dir(&path);
        fs::write(format!("{path}/ruda-compiler.ir.txt"), format!("{}", opt)).unwrap();
        fs::write(
            format!("{path}/ruda-compiler.ir.dot"),
            format!("{}", opt.dot_viz()),
        )
        .unwrap();
    }
}
