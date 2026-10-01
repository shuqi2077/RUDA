use ruda_core::backtrace::BackTrace;
use ruda_core::ir::DeviceProperties;

use crate::runtime::{
    id::KernelId,
    server::{RudaDim, LaunchError, ResourceLimitError},
};

/// Validate the ruda dim of a kernel fits within the hardware limits
pub fn validate_ruda_dim(
    properties: &DeviceProperties,
    kernel_id: &KernelId,
) -> Result<(), LaunchError> {
    let requested = kernel_id.ruda_dim;
    let max: RudaDim = properties.hardware.max_ruda_dim.into();
    if !max.can_contain(requested) {
        Err(ResourceLimitError::RudaDim {
            requested: requested.into(),
            max: max.into(),
            backtrace: BackTrace::capture(),
        }
        .into())
    } else {
        Ok(())
    }
}

/// Validate the total units of a kernel fits within the hardware limits
pub fn validate_units(
    properties: &DeviceProperties,
    kernel_id: &KernelId,
) -> Result<(), LaunchError> {
    let requested = kernel_id.ruda_dim.num_elems();
    let max = properties.hardware.max_units_per_ruda;
    if requested > max {
        Err(ResourceLimitError::Units {
            requested,
            max,
            backtrace: BackTrace::capture(),
        }
        .into())
    } else {
        Ok(())
    }
}
