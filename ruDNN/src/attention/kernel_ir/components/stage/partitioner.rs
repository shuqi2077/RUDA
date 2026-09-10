use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

use crate::attention::kernel_ir::components::global::simple::AttentionWriter;

#[ruda]
/// Defines how the stage is partitioned among compute primitives (e.g., units or planes).
/// Controls global writeback and compute indexing.
pub trait AttentionPartitioner: Send + Sync + 'static {
    type Writer<ES: Float, ESS: Size, EG: Float, EGS: Size>: AttentionWriter<ES, ESS, EG, EGS>;

    fn seq_q_index() -> u32;
}
