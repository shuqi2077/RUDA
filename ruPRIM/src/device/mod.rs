//! Device-wide primitives using Ruda IR and device-resident intermediate data.

use ruda_core::tensor::{TensorMetadata, element::Element};
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement};

pub mod scan;
pub mod sort;
pub mod transform;
pub mod select;
pub mod radix;
pub mod merge;
pub mod segmented;
pub mod reduce;
pub mod find;
pub mod iteration;
pub mod run_length;
pub mod segments;
pub mod adjacent;
pub mod histogram;
pub mod segmented_sort;
pub mod topk;
pub mod arg_reduce;
pub mod partition;
pub mod copy;
pub mod memcpy;
pub mod access;
pub mod record;
pub mod double_buffer;
pub mod typed;

#[derive(Debug, thiserror::Error)]
pub enum RudaPrimitiveError {
    #[error("invalid primitive configuration: {0}")]
    Configuration(&'static str),
    #[error("tensor dtype does not match the primitive's element type")]
    Dtype,
    #[error("input lengths differ")]
    Length,
    #[error("inputs belong to different devices")]
    Device,
}

pub(crate) fn check_type<R: Runtime, T: TensorElement>(
    input: &RudaTensor<R>,
) -> Result<(), RudaPrimitiveError> {
    if input.dtype != <T as Element>::dtype() {
        return Err(RudaPrimitiveError::Dtype);
    }
    Ok(())
}

pub(crate) fn empty_like<R: Runtime>(input: &RudaTensor<R>) -> RudaTensor<R> {
    empty_device_dtype(input.client.clone(), input.device.clone(), input.shape(), input.dtype)
}

pub(crate) fn scan_threads<R: Runtime, T: TensorElement>(
    input: &RudaTensor<R>,
    threads: u32,
) -> Result<(), RudaPrimitiveError> {
    let hardware = &input.client.properties().hardware;
    if threads < 2 || threads > hardware.max_units_per_ruda || threads > hardware.max_ruda_dim.0 {
        return Err(RudaPrimitiveError::Configuration("scan threads must be in 2..=device block limit"));
    }
    if threads as usize * core::mem::size_of::<T>() > hardware.max_shared_memory_size {
        return Err(RudaPrimitiveError::Configuration("scan scratch exceeds device shared memory"));
    }
    Ok(())
}
