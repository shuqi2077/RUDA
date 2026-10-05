use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::Swizzle;
use ruda_kernel::library::tensor::View;
use ruda_kernel::library::tensor::layout::Coords2d;
use rublas::kernel_ir::components::global::memory::GlobalMemoryConfig;
use ruda_kernel::tiling::tile::StridedTile;

use crate::attention::kernel_ir::{
    components::stage::AttentionPartitioner,
    definition::attention_types::{QG, QGS},
    definition::{AttentionPrecision, AttentionTileSize},
};

#[derive(RudaType)]
pub struct QueryReader<AP: AttentionPrecision> {
    query: View<Vector<QG<AP>, QGS<AP>>, Coords2d>,
    #[ruda(comptime)]
    gmem_config: GlobalMemoryConfig,
}

#[ruda]
impl<AP: AttentionPrecision> QueryReader<AP> {
    pub fn new(
        stage_q_offset: u32,
        query: View<Vector<QG<AP>, QGS<AP>>, Coords2d>,
        #[comptime] gmem_config: GlobalMemoryConfig,
    ) -> Self {
        let query = query.slice((stage_q_offset, 0), query.shape());

        QueryReader::<AP> { query, gmem_config }
    }

    pub fn get_tile<P: AttentionPartitioner>(
        &self,
        tile: Coords2d,
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] partition_seq_q: u32,
        #[comptime] partition_head_dim: u32,
    ) -> StridedTile<QG<AP>, QGS<AP>> {
        let (row_in_partition, col) = tile;

        let row = row_in_partition + P::seq_q_index() * partition_seq_q;

        let vector_size = self.gmem_config.vector_size.comptime() as u32;

        let slice = self
            .query
            .slice(
                (row * tile_size.seq_q, col * tile_size.head_dim),
                (tile_size.seq_q, tile_size.head_dim).runtime(),
            )
            .to_linear_slice();

        let start = 0;
        let vectors_per_tile = tile_size.seq_q * tile_size.head_dim / vector_size;
        let end = start + vectors_per_tile;
        let vectors_per_partition_row = partition_head_dim * tile_size.head_dim / vector_size;

        StridedTile::<QG<AP>, QGS<AP>>::new_strided(
            slice,
            start,
            end,
            vectors_per_partition_row,
            Swizzle::none(),
            self.gmem_config.matrix_layout,
        )
    }

    pub fn head_dim(&self) -> u32 {
        self.query.shape().1
    }

    pub fn get_staged_tile<P: AttentionPartitioner>(
        &self,
        tile: Coords2d,
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] partition_seq_q: u32,
        #[comptime] plane_dim: u32,
        #[comptime] num_planes: u32,
    ) -> StridedTile<QG<AP>, QGS<AP>> {
        #[comptime]
        let vector_size = self.gmem_config.vector_size;
        #[comptime]
        let vectors_per_row = tile_size.head_dim / vector_size as u32;
        #[comptime]
        let vectors_per_tile = tile_size.seq_q * vectors_per_row;
        let start = UNIT_POS_Y * vectors_per_tile;
        let mut storage = SharedMemory::<Vector<QG<AP>, QGS<AP>>>::new(
            (vectors_per_tile * num_planes) as usize,
        );
        let mut data = storage.slice_mut(start as usize, (start + vectors_per_tile) as usize);
        let row = (tile.0 + P::seq_q_index() * partition_seq_q) * tile_size.seq_q;
        let col = tile.1 * tile_size.head_dim;
        let mut index = UNIT_POS_X;
        while index < vectors_per_tile {
            data[index as usize] = self.query.read_checked((
                row + index / vectors_per_row,
                col + (index % vectors_per_row) * vector_size as u32,
            ));
            index += plane_dim;
        }
        sync_ruda();
        StridedTile::<QG<AP>, QGS<AP>>::new_strided(
            data.to_slice(),
            0,
            vectors_per_tile,
            vectors_per_row,
            Swizzle::none(),
            self.gmem_config.matrix_layout,
        )
    }
}
