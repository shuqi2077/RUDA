use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

use crate::attention::kernel_ir::components::stage::SharedPartitionAttentionConfig;
use crate::attention::kernel_ir::components::{
    global::simple::UnitAttentionWriter,
    stage::{partition_attention::PartitionAttention, partitioner::AttentionPartitioner},
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitPartitionStageConfig {
    pub shared: SharedPartitionAttentionConfig,
}

pub type UnitPartitionAttention<AP, SK, SV, SO> =
    PartitionAttention<AP, SK, SV, SO, UnitPartitioner>;

pub struct UnitPartitioner {}

#[ruda]
impl AttentionPartitioner for UnitPartitioner {
    type Writer<ES: Float, ESS: Size, EG: Float, EGS: Size> = UnitAttentionWriter<ES, ESS, EG, EGS>;

    fn seq_q_index() -> u32 {
        UNIT_POS
    }
}
