use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::RudaDim;

use crate::attention::kernel_ir::components::{batch::BatchAttentionConfig, global::GlobalAttentionConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SimpleBatchConfig<G: GlobalAttentionConfig> {
    global_config: G,
}

impl<G: GlobalAttentionConfig> BatchAttentionConfig for SimpleBatchConfig<G> {
    type GlobalConfig = G;

    fn global_config(&self) -> Self::GlobalConfig {
        self.global_config
    }

    fn ruda_dim(&self) -> RudaDim {
        self.global_config.ruda_dim()
    }
}

impl<G: GlobalAttentionConfig> SimpleBatchConfig<G> {
    pub fn new(global_config: G) -> Self {
        Self { global_config }
    }
}
