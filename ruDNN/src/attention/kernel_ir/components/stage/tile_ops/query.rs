use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tiling::tile::StridedTile;

use crate::attention::kernel_ir::{
    definition::AttentionPrecision,
    definition::attention_types::QG,
    {components::tile::TileAttention, definition::attention_types::QGS},
};

#[derive(RudaType)]
/// Query input to the Tile Attention
pub struct QueryTile<AP: AttentionPrecision, TA: TileAttention<AP>> {
    pub fragment: TA::Query,
}

#[ruda]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> QueryTile<AP, TA> {
    pub fn new(#[comptime] config: TA::Config) -> QueryTile<AP, TA> {
        QueryTile::<AP, TA> {
            fragment: TA::allocate_query(config),
        }
    }

    /// Loads the query data into the fragment
    pub fn update(&mut self, tile: &StridedTile<QG<AP>, QGS<AP>>) {
        TA::load_query(tile, &mut self.fragment)
    }
}
