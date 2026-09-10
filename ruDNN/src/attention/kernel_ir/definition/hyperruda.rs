//! Attention-specific interpretation of the generic [RudaMapping] from `ruda-kernel::tiling`.
//!
//! Attention has 2D problem-space axes: `(seq_q_tile, batch_heads)` where
//! `batch_heads = batch * num_heads`. The [HyperrudaBlueprint], [RudaCountPlan]
//! and [RudaMapping] types come directly from `ruda-kernel::tiling`; this module only
//! adds the attention-specific `(seq_q, batch_heads)` mapper.

use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

pub use ruda_kernel::tiling::ruda_count::{
    RudaCountPlan, RudaMapping, RudaMappingLaunch, HyperrudaBlueprint, ruda_mapping_launch,
};

#[ruda]
/// Reads the ruda position as attention `(seq_q_index, batch_heads_index)` coordinates.
///
/// The `batch_heads_index` spans `batch * num_heads`; the third axis is unused.
pub fn ruda_pos_to_q_batch_heads(ruda_mapping: &RudaMapping) -> (u32, u32) {
    let (seq_q, batch_heads, _) = ruda_mapping.ruda_pos_to_xyz();
    (seq_q, batch_heads)
}
