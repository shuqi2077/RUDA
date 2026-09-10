use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::r#virtual::VirtualTensor;
use std::marker::PhantomData;

use crate::attention::kernel_ir::components::{
    batch::{BatchAttention, BatchAttentionConfig, simple::config::SimpleBatchConfig},
    global::{GlobalAttention, GlobalAttentionConfig as _},
    stage::StageAttentionConfig as _,
};
use crate::attention::kernel_ir::{
    definition::attention_types::*,
    definition::{AttentionPrecision, RudaMapping, ruda_pos_to_q_batch_heads},
};

pub struct SimpleBatchAttention<AP: AttentionPrecision, GA: GlobalAttention<AP>> {
    _phantom: PhantomData<(AP, GA)>,
}

#[ruda]
impl<GA: GlobalAttention<AP>, AP: AttentionPrecision> BatchAttention<AP>
    for SimpleBatchAttention<AP, GA>
{
    type Config = SimpleBatchConfig<GA::Config>;

    fn execute(
        query: VirtualTensor<QG<AP>, QGS<AP>>,
        key: VirtualTensor<KG<AP>, KGS<AP>>,
        value: VirtualTensor<VG<AP>, VGS<AP>>,
        mask: ComptimeOption<VirtualTensor<MSK<AP>, MSKS<AP>>>,
        out: VirtualTensor<OG<AP>, OGS<AP>, ReadWrite>,
        ruda_mapping: RudaMapping,
        #[comptime] config: Self::Config,
    ) {
        #[allow(clippy::collapsible_if)]
        if ruda_mapping.can_yield_extra_rudas {
            if RUDA_POS >= ruda_mapping.num_valid_rudas() {
                terminate!();
            }
        }

        let global_config = config.global_config();
        let (q_index, batch_index) = ruda_pos_to_q_batch_heads(&ruda_mapping);

        let stage_q_offset = q_index * global_config.stage_config().elements_in_stage_seq_q();

        // Assume [batch, num_heads, seq_*, head_dim] layout
        let seq_q = query.shape(2) as u32;
        let seq_kv = key.shape(2) as u32;

        GA::execute(
            GA::init_query_reader(batch_index, stage_q_offset, query, global_config),
            GA::init_key_reader(batch_index, key, global_config),
            GA::init_value_reader(batch_index, value, global_config),
            GA::init_mask_reader(batch_index, stage_q_offset, mask, seq_kv, global_config),
            GA::init_writer(batch_index, stage_q_offset, out, global_config),
            seq_q,
            seq_kv,
            config.global_config(),
        )
    }
}
