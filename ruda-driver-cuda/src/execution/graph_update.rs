//! Scalar-only graph update validation. No GPU calls or permissive rebinding.
use crate::{execution::graph::{error, view_key}, graph::GraphDispatch};
use ruda::runtime::{id::KernelId, server::{Binding, ExecutionMode, RudaCount, ServerError}};

#[derive(Debug)]
pub(super) struct NodeSignature {
    kernel: KernelId,
    grid: (u32, u32, u32),
    buffers: Vec<Binding>,
    scalar_words: usize,
    dynamic_offset: usize,
    info: Vec<u64>,
}

/// Validate before indexing the packed scalar/metadata buffer. Identical bit
/// patterns are a no-op (including signed zero/NaN payloads, without float ==).
pub(super) fn check_info(old: &[u64], old_scalars: usize, old_dynamic: usize,
    new: &[u64], new_scalars: usize, new_dynamic: usize) -> Result<bool, ServerError>
{
    if old_scalars > old_dynamic || old_dynamic > old.len()
        || new_scalars > new_dynamic || new_dynamic > new.len() {
        return Err(error("invalid scalar/metadata layout"));
    }
    if old_scalars != new_scalars || old_dynamic != new_dynamic || old.len() != new.len() {
        return Err(error("graph update changes scalar/metadata layout"));
    }
    if old[old_scalars..] != new[new_scalars..] {
        return Err(error("graph update changes shape, stride, length or dynamic metadata"));
    }
    Ok(old[..old_scalars] != new[..new_scalars])
}
impl NodeSignature {
    pub fn new(dispatch: &GraphDispatch) -> Result<Self, ServerError> {
        crate::graph::validate_dispatch(&dispatch.count, &dispatch.arguments)?;
        let RudaCount::Static(x,y,z) = &dispatch.count else { unreachable!() };
        let mut kernel = dispatch.task.id(); kernel.mode(ExecutionMode::Checked);
        let args = &dispatch.arguments;
        check_info(&args.info.data, dispatch.scalar_words, args.info.dynamic_metadata_offset,
            &args.info.data, dispatch.scalar_words, args.info.dynamic_metadata_offset)?;
        Ok(Self { kernel, grid:(*x,*y,*z), buffers:args.buffers.clone(), scalar_words:dispatch.scalar_words,
            dynamic_offset:args.info.dynamic_metadata_offset, info:args.info.data.clone() })
    }
    pub fn validate(&self, dispatch: &GraphDispatch) -> Result<bool, ServerError> {
        crate::graph::validate_dispatch(&dispatch.count, &dispatch.arguments)?;
        let mut kernel = dispatch.task.id(); kernel.mode(ExecutionMode::Checked);
        let RudaCount::Static(x,y,z) = &dispatch.count else { unreachable!() };
        if kernel != self.kernel || (*x,*y,*z) != self.grid {
            return Err(error("graph update changes kernel, specialization, block or grid"));
        }
        let args = &dispatch.arguments;
        if self.buffers.len() != args.buffers.len() || self.buffers.iter().zip(&args.buffers)
            .any(|(a,b)| view_key(a) != view_key(b)) {
            return Err(error("graph update changes allocation identity, view or origin stream"));
        }
        check_info(&self.info,self.scalar_words,self.dynamic_offset,
            &args.info.data,dispatch.scalar_words,args.info.dynamic_metadata_offset)
    }
    pub fn commit(&mut self, data: &[u64]) { self.info[..self.scalar_words].copy_from_slice(&data[..self.scalar_words]); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn changed_scalar_is_allowed() { assert!(check_info(&[1,8,9],1,2,&[2,8,9],1,2).unwrap()); }
    #[test] fn identical_bits_are_a_noop() { assert!(!check_info(&[0x7ff8000000000001,8],1,2,&[0x7ff8000000000001,8],1,2).unwrap()); }
    #[test] fn signed_zero_bits_are_not_collapsed() { assert!(check_info(&[0,8],1,2,&[1u64<<63,8],1,2).unwrap()); }
    #[test] fn static_metadata_changes_rejected() { assert!(check_info(&[1,8,9],1,2,&[2,7,9],1,2).is_err()); }
    #[test] fn dynamic_metadata_changes_rejected() { assert!(check_info(&[1,8,9],1,2,&[2,8,7],1,2).is_err()); }
    #[test] fn scalar_boundary_changes_rejected() { assert!(check_info(&[1,8,9],1,2,&[2,8,9],2,2).is_err()); }
    #[test] fn offset_changes_rejected() { assert!(check_info(&[1,8,9],1,2,&[2,8,9],1,3).is_err()); }
    #[test] fn length_changes_rejected() { assert!(check_info(&[1,8],1,2,&[2,8,9],1,2).is_err()); }
    #[test] fn invalid_boundaries_rejected_without_slice_panic() {
        for (scalars,dynamic) in [(4,2),(2,4),(usize::MAX,usize::MAX)] {
            assert!(check_info(&[1,8,9],scalars,dynamic,&[1,8,9],1,2).is_err());
            assert!(check_info(&[1,8,9],1,2,&[1,8,9],scalars,dynamic).is_err());
        }
    }
    #[test] fn empty_and_no_scalar_nodes_are_noops() {
        assert!(!check_info(&[],0,0,&[],0,0).unwrap());
        assert!(!check_info(&[5,8],0,2,&[5,8],0,2).unwrap());
        assert!(check_info(&[5,8],0,2,&[5,9],0,2).is_err());
    }
}
