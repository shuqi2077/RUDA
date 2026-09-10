use ruda_kernel::dsl::prelude::*;
use ruda_kernel::dsl as kernel_dsl;

use rublas::kernel_ir::components::global::{GlobalWriterConfig, PartitionedStage, WriteEventListener};

mod plane;
mod unit;

use ruda_kernel::library::tensor::View;
use ruda_kernel::library::tensor::layout::Coords2d;
pub use plane::*;
pub use unit::*;

use crate::attention::kernel_ir::components::stage::StageAttentionConfig;

#[ruda]
pub trait AttentionWriter<ES: Numeric, ESS: Size, EG: Numeric, EGS: Size>:
    WriteEventListener
{
    fn init<S: StageAttentionConfig>(
        global: View<Vector<EG, EGS>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self;

    fn stage(&mut self) -> PartitionedStage<ES, ESS>;
}
