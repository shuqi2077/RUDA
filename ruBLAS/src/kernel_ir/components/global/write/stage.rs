use ruda_kernel::dsl as kernel_dsl;
use crate::kernel_ir::components::stage::{
    ContiguousTilingLayout, RowMajorTilingOrder, Stage, StageFamily, StridedStageMemory,
    TilingLayout,
};
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::layout::Coords2d;
use ruda_kernel::tiling::{
    stage::StageMemoryConfig,
    tile::{SharedTile, StridedTile, Tile, TileScope},
};

pub type WriteTiling = ContiguousTilingLayout<RowMajorTilingOrder>;

pub struct PartitionedStageFamily;

impl StageFamily<ReadWrite> for PartitionedStageFamily {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout> = PartitionedStage<ES, NS>;
}

#[derive(RudaType, Clone, Copy)]
/// Layoutless stage for current writers. Tile only depends on the unit index, not the out tile.
pub struct PartitionedStage<ES: Numeric, NS: Size> {
    /// Underlying shared memory
    _smem: SharedMemory<Vector<ES, NS>>,
    pub unit_tile: StridedTile<ES, NS, ReadWrite>,
}

#[ruda]
impl<ES: Numeric, NS: Size> PartitionedStage<ES, NS> {
    /// Instantiate a new stage memory for the given identifier
    pub fn new(
        unit_pos: Coords2d,
        #[comptime] config: StageMemoryConfig,
    ) -> PartitionedStage<ES, NS> {
        let config = comptime![StageMemoryConfig {
            tiles_per_partition_along_row: 1,
            tiles_per_partition_along_col: 1,
            ..config
        }];

        // Needs to be 16-byte aligned for `stmatrix`
        let inner = StridedStageMemory::<ES, NS, WriteTiling>::new_aligned(16usize, config);

        let tile = inner.get_tile_mut(unit_pos);

        PartitionedStage::<ES, NS> {
            _smem: inner.smem,
            unit_tile: tile,
        }
    }
}

#[ruda]
impl<ES: Numeric, NS: Size> Stage<ES, ReadWrite> for PartitionedStage<ES, NS> {
    fn tile<Sc: TileScope>(this: &Self, _tile: Coords2d) -> Tile<ES, Sc, ReadWrite> {
        Tile::new_SharedMemory(SharedTile::wrap::<NS>(this.unit_tile))
    }
}
