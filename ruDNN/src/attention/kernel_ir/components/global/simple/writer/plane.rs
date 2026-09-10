use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::View;
use ruda_kernel::library::tensor::layout::Coords2d;
use ruda_kernel::dsl as kernel_dsl;
use rublas::kernel_ir::components::global::{
    GlobalWriterConfig, PartitionedStage, WriteEvent, WriteEventExpand, WriteEventListener,
    plane_write,
    read::tiled::{TiledCoords, TiledLayout},
};
use ruda_kernel::tiling::StageIdent;

use crate::attention::kernel_ir::components::{
    global::simple::{AttentionWriter, AttentionWriterExpand},
    stage::{AttentionPartitioner, StageAttentionConfig, plane::PlanePartitioner},
};

#[derive(RudaType)]
pub struct PlaneAttentionWriter<ES: Numeric, ESS: Size, EO: Numeric, EOS: Size> {
    global: View<Vector<EO, EOS>, TiledCoords, ReadWrite>,
    stage: PartitionedStage<ES, ESS>,

    #[ruda(comptime)]
    config: GlobalWriterConfig,
}

#[ruda]
impl<ES: Numeric, ESS: Size, EG: Numeric, EGS: Size> PlaneAttentionWriter<ES, ESS, EG, EGS> {}

#[ruda]
impl<ES: Numeric, ESS: Size, EG: Numeric, EGS: Size> WriteEventListener
    for PlaneAttentionWriter<ES, ESS, EG, EGS>
{
    fn on_event(this: &mut Self, event: WriteEvent) {
        #[allow(clippy::single_match)]
        match event {
            WriteEvent::TileStored { tile } => plane_write::<ES, ESS, EG, EGS>(
                &mut this.global,
                &this.stage.unit_tile,
                tile,
                this.config.plane_dim,
                this.config.comptime().smem_config.elements_per_tile(),
            ),
            _ => {}
        }
    }
}

#[ruda]
impl<ES: Numeric, ESS: Size, EG: Numeric, EGS: Size> AttentionWriter<ES, ESS, EG, EGS>
    for PlaneAttentionWriter<ES, ESS, EG, EGS>
{
    fn init<S: StageAttentionConfig>(
        global: View<Vector<EG, EGS>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        let stage =
            PartitionedStage::new((PlanePartitioner::seq_q_index(), 0u32), config.smem_config);

        PlaneAttentionWriter::<ES, ESS, EG, EGS> {
            global: global.view_mut(TiledLayout::new(StageIdent::Out, config.smem_config)),
            stage,
            config,
        }
    }

    fn stage(&mut self) -> PartitionedStage<ES, ESS> {
        self.stage
    }
}
