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
}
