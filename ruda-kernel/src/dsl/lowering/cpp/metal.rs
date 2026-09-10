use alloc::{vec, vec::Vec};
use ruda_core::ir::Visibility;
use ruda::runtime::server::KernelArguments;
use ruda_compiler::cpp::shared::MslComputeKernel;

pub fn bindings(
    repr: &MslComputeKernel,
    args: &KernelArguments,
) -> (Vec<Visibility>, Option<Visibility>, bool) {
    let mut bindings: Vec<Visibility> = vec![];
    // must be in the same order as the compilation order: inputs, outputs and named
    for b in repr.buffers.iter() {
        bindings.push(b.vis);
    }
    let info = (!args.info.data.is_empty()).then_some(Visibility::Read);
    let uniform = args.info.dynamic_metadata_offset >= args.info.data.len();
    (bindings, info, uniform)
}
