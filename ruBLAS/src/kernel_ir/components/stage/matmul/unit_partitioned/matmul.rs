use ruda_kernel::dsl as kernel_dsl;
use crate::kernel_ir::{
    components::{
        global::{PlaneFlowPartition, PlaneFlowPartitionRule},
        stage::matmul::partitioned_matmul::{PartitionedStageMatmul, StagePartitioner},
    },
    definition::MatmulTypes,
};
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::layout::Coords2d;
use ruda_kernel::tiling::tile::Unit;

use crate::kernel_ir::components::stage::matmul::partition::SharedPartitionMatmulConfig;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the unit partitioned stage matmul
pub struct UnitPartitionedStageConfig {
    pub shared: SharedPartitionMatmulConfig,
}

impl UnitPartitionedStageConfig {
    pub fn from_shared_partition_config(shared: SharedPartitionMatmulConfig) -> Self {
        Self { shared }
    }
}

#[allow(type_alias_bounds)]
/// [PartitionedStageMatmul] partitioned across units
pub type UnitMatmul<MP: MatmulTypes, StageLhs, StageRhs, StageAcc, StageOut> =
    PartitionedStageMatmul<MP, StageLhs, StageRhs, StageAcc, StageOut, UnitPartitioner>;

/// Defines how to partition across units
pub struct UnitPartitioner {}

#[ruda]
impl StagePartitioner for UnitPartitioner {
    type Scope = Unit;

    fn coordinates(
        #[comptime] role_rule_config: PlaneFlowPartitionRule,
        #[comptime] plane_dim: u32,
        #[comptime] num_partitions_col: u32,
    ) -> Coords2d {
        let plane_id = PlaneFlowPartition::new(role_rule_config).compute_index();

        let absolute_index = UNIT_POS_X + plane_dim * plane_id;

        (
            absolute_index / num_partitions_col,
            absolute_index % num_partitions_col,
        )
    }
}
