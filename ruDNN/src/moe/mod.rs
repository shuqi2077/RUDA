mod dispatch;
mod experts;
mod kernels;
mod routing;

pub use dispatch::DispatchedTokens;
pub use experts::SwiGluExperts;
pub use routing::{RoutingOptions, RoutingPlan, route};

use ruda_core::{
    device::Device,
    tensor::{DType, Shape},
};
use ruda_kernel::{
    dsl::Runtime,
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype},
};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeError(pub &'static str);

impl fmt::Display for MoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for MoeError {}

impl From<rublas::tensor_grouped::GroupedMatmulError> for MoeError {
    fn from(error: rublas::tensor_grouped::GroupedMatmulError) -> Self {
        Self(error.0)
    }
}

fn elements(dimensions: &[usize]) -> Result<usize, MoeError> {
    if dimensions.iter().any(|&dim| dim > u32::MAX as usize) {
        return Err(MoeError("MoE dimension exceeds U32 indexing"));
    }
    dimensions
        .iter()
        .try_fold(1usize, |size, &dim| size.checked_mul(dim))
        .filter(|&size| size <= u32::MAX as usize)
        .ok_or(MoeError("MoE tensor exceeds U32 indexing"))
}

fn same_device<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>) -> Result<(), MoeError> {
    if a.device.to_id() != b.device.to_id() {
        return Err(MoeError("MoE operands must share a device"));
    }
    Ok(())
}

fn float_tensor<R: Runtime>(tensor: &RudaTensor<R>) -> Result<(), MoeError> {
    if tensor.qparams.is_some() || !matches!(tensor.dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(MoeError(
            "MoE requires non-quantized F32, F16 or BF16 tensors",
        ));
    }
    Ok(())
}

fn empty<R: Runtime>(like: &RudaTensor<R>, shape: impl Into<Shape>, dtype: DType) -> RudaTensor<R> {
    empty_device_contiguous_dtype(
        like.client.clone(),
        like.device.clone(),
        shape.into(),
        dtype,
    )
}

#[cfg(test)]
mod tests;
