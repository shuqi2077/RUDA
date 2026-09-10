use ruda_kernel::dsl as kernel_dsl;
use crate::kernel_ir::{
    components::{
        global::{
            PlaneFlowPartitionRule, WriteEventListener, WriteTiling, memory::GlobalMemoryConfig,
        },
        stage::{Stage, StageFamily},
    },
    definition::MatrixTypes,
};
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::View;
use ruda_kernel::library::tensor::layout::Coords2d;
use ruda_kernel::tiling::stage::StageMemoryConfig;

pub trait GlobalWriterFamily: 'static + Send + Sync {
    type Stage: StageFamily<ReadWrite>;
    type Writer<IP: MatrixTypes>: GlobalWriter<
            IP,
            Stage = <Self::Stage as StageFamily<ReadWrite>>::Stage<
                IP::Stage,
                IP::StageSize,
                WriteTiling,
            >,
        >;
}

#[ruda]
/// Responsible of writing the accumulated stage matmul output
/// to global memory
pub trait GlobalWriter<IP: MatrixTypes>:
    WriteEventListener + RudaType + 'static + Send + Sync
{
    /// Tile stage that stores the data for this writer
    type Stage: Stage<IP::Stage, ReadWrite>;

    /// Init this writer from a global tensor and config
    fn init(
        tensor: View<Vector<IP::Global, IP::GlobalSize>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self;

    /// Stage used by this writer
    fn stage(this: &Self) -> Self::Stage;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GlobalWriterConfig {
    pub gmem_config: GlobalMemoryConfig,
    pub smem_config: StageMemoryConfig,
    pub plane_flow_partition_rule: PlaneFlowPartitionRule,
    pub plane_dim: u32,
}
