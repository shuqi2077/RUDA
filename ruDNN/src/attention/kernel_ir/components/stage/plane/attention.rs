use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

use crate::attention::kernel_ir::components::stage::SharedPartitionAttentionConfig;
use crate::attention::kernel_ir::components::{
    global::simple::PlaneAttentionWriter,
    stage::{partition_attention::PartitionAttention, partitioner::AttentionPartitioner},
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PlanePartitionStageConfig {
    pub shared: SharedPartitionAttentionConfig,
}

pub type PlanePartitionAttention<AP, SK, SV, SO> =
    PartitionAttention<AP, SK, SV, SO, PlanePartitioner>;

pub struct PlanePartitioner {}

#[ruda]
impl AttentionPartitioner for PlanePartitioner {
    type Writer<ES: Float, ESS: Size, EG: Float, EGS: Size> =
        PlaneAttentionWriter<ES, ESS, EG, EGS>;

    fn seq_q_index() -> u32 {
        UNIT_POS_Y
    }
}
