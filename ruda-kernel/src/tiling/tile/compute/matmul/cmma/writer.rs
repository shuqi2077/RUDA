use crate::dsl::prelude::*;

use crate::tiling::{as_cmma_layout, tile::data::StridedTile};

/// Writer using the cmma store function.
#[derive(RudaType)]
pub struct CmmaStageWriter {}

#[ruda]
impl CmmaStageWriter {
    pub fn store_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &mut StridedTile<V, N, ReadWrite>,
        fragment: &cmma::Matrix<E>,
    ) {
        let layout = as_cmma_layout(tile.layout);
        let stride = tile.unvectorized_stride();
        let mut slice = tile.as_slice_mut();
        cmma::store(&mut slice, fragment, stride, layout);
    }
}
