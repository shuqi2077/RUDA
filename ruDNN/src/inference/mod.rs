#![allow(unsafe_code)]
//! Native Rust inference operators. Planning uploads only scheduler metadata;
//! tensor data and all numerical work remain on the device.
//!
//! This module is not a binding to FlashInfer. See `docs/en/flashinfer.md` for the
//! versioned operator scope and device-buffer contracts.
pub mod activation;
pub mod attention;
pub mod cascade;
pub mod config;
pub mod experts;
pub mod fused;
pub mod indexer;
pub mod linear;
pub mod mamba;
pub mod masks;
pub mod metadata;
pub mod mhc;
pub mod msa;
pub mod norm;
pub mod nvfp4_attention;
pub mod page;
pub mod paged;
pub mod peer_candidates;
#[cfg(feature = "inference-cuda")]
pub mod peer_group;
pub mod peer_moe;
pub mod pre_attention;
pub mod recurrence;
pub mod recurrence_paged;
pub mod replay;
pub mod rope;
pub mod sage;
pub mod selective;
pub mod sharded_indexer;
pub mod sparse;
pub mod state_storage;
pub mod trace;
pub mod training;
#[cfg(all(
    feature = "inference-autotune",
    any(
        target_os = "windows",
        target_os = "linux",
        target_os = "macos",
        target_os = "android"
    )
))]
pub mod tuning;
pub mod video;

use ruda_core::{
    device::Device,
    tensor::{DType, Shape},
};
use ruda_kernel::{
    dsl::prelude::*,
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceError(pub &'static str);
impl core::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for InferenceError {}

pub(super) fn alloc<R: Runtime>(
    like: &RudaTensor<R>,
    shape: impl Into<Shape>,
    dtype: DType,
) -> RudaTensor<R> {
    empty_device_contiguous_dtype(
        like.client.clone(),
        like.device.clone(),
        shape.into(),
        dtype,
    )
}
pub(super) fn check<R: Runtime>(
    t: &RudaTensor<R>,
    like: &RudaTensor<R>,
) -> Result<(), InferenceError> {
    check_view(t, like)?;
    if !t.is_contiguous() {
        return Err(InferenceError("expected tightly contiguous tensor"));
    }
    Ok(())
}
pub(super) fn check_view<R: Runtime>(
    t: &RudaTensor<R>,
    like: &RudaTensor<R>,
) -> Result<(), InferenceError> {
    if t.qparams.is_some()
        || t.device.to_id() != like.device.to_id()
        || !t.client.same_execution_queue(&like.client)
        || t.meta.num_elements() > u32::MAX as usize
    {
        return Err(InferenceError(
            "expected same-queue contiguous U32-indexed unquantized tensors",
        ));
    }
    Ok(())
}
/// Reusable stride-aware device packing. A dense source is borrowed directly;
/// otherwise one destination is allocated during preparation and reused.
pub struct PreparedPack<R: Runtime> {
    source: RudaTensor<R>,
    pub output: RudaTensor<R>,
    copy: bool,
}
impl<R: Runtime> PreparedPack<R> {
    pub fn prepare(source: &RudaTensor<R>) -> Result<Self, InferenceError> {
        check_view(source, source)?;
        let copy = !source.is_contiguous();
        let output = if copy {
            alloc(source, source.meta.shape().clone(), source.dtype)
        } else {
            source.clone()
        };
        Ok(Self {
            source: source.clone(),
            output,
            copy,
        })
    }
    pub unsafe fn run(&self) {
        if self.copy {
            ruda_kernel::library::tensor::copy_into(
                &self.source.client,
                self.source.clone().binding(),
                self.output.clone().binding(),
                self.source.dtype.into(),
            );
        }
    }
    /// Commit packed mutations back to the original strided view. The original
    /// allocation must be exclusively owned by this operation during the copy.
    pub unsafe fn write_back(&self) {
        if self.copy {
            ruda_kernel::library::tensor::copy_into(
                &self.source.client,
                self.output.clone().binding(),
                self.source.clone().binding(),
                self.source.dtype.into(),
            );
        }
    }
}
pub(super) fn float<R: Runtime>(t: &RudaTensor<R>) -> Result<(), InferenceError> {
    if !matches!(t.dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(InferenceError("expected F32/F16/BF16 tensor"));
    }
    Ok(())
}
pub(super) fn plane<R: Runtime>(
    like: &RudaTensor<R>,
    rows: usize,
    heads: usize,
) -> Result<u32, InferenceError> {
    let hw = &like.client.properties().hardware;
    if !matches!(hw.plane_size_max, 32 | 64)
        || rows > hw.max_ruda_count.0 as usize
        || heads > hw.max_ruda_count.1 as usize
    {
        return Err(InferenceError(
            "requires full 32/64-lane planes and legal grid dimensions",
        ));
    }
    Ok(hw.plane_size_max)
}
